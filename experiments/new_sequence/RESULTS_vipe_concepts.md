# ViPE's concepts in our chain — the trial and its numbers (2026-11-22 / 23)

HIS VERDICT (USER 2026-11-23): *"el validado sigue ganando, queda el validado … ninguno le gano al validado, no hay
otra"* / *"debe quedar el pipeline original solo con la correccion de oscuros, y el resto fuera del pipeline, si
queres en una rama aparte solo por cuestiones de pruebas futuras, pero guarda los numeros obtenidos, y nos quedamos
con lo que teniamos que funciono perfecto"* / *"yo dejaria F2 apagado pointdit afuera en el principal"* / *"con lo que
corrimos ya es suficiente evidencia … el resto campo B, y moge 3 a la rama, ok!"* / *"acordate de poner F5 configurado
teniendo en cuenta que no esta F2"*.

## The main pipeline after it

The validated pipeline (CLAUDE.md §0; its reconstruction code = commit 25636e2) with three changes, nothing else:

1. **The dark frames** — `intake.quality.luma_lo` 40 → 0 and `clip_frac_max` 0.25 → 1.0: no frame is rejected by
   its exposure (pccr 2026-08-24's dark turn, video 6274–6403, had no keyframe and its floor was published 36 cm low).
2. **F2 off** — `reconstruction.precision.gauge.enabled: false`. F5's one input from F2, the relative tolerance of
   its position anchors (`gauge.json`, read by `refine._gauge_sigma`), is the tie's own error: the median over the
   cube graph's seams of residual / the distance it was measured at (`precision/tie_tolerance.py`), written by the
   chain as `output/gauge.json` before the steps' stamps, for a chain that runs F5. F5 starts from the tie's poses.
3. **PointDiT off, not deleted** — `reconstruction.precision.mono_detail.enabled: false`.

That exact combination has not run as such: the runs closest to it (4 and 5 below) also had F5's metric anchor and
F6's field B. His word: enough evidence.

## The trial's code — branch `trial/vipe-concepts` (20ac416)

- **F5's metric anchor** (MetricTrail's concept, `precision/ba_torch.py`, `precision/refine.py`): every fit landmark
  tied, under the same Huber loss, to Omega's depth in its nearest keyframe; a log-scale UNKNOWN per Omega cube in the
  Schur system; the weight κ = σ_px / σ_log measured at the solution and re-measured (R0 solved again) until it moves
  ≤ 1.1 × its cluster-bootstrap se, at most 10 rounds (`refine.metric_anchor`, `refine.metric_anchor_reweights`).
- **F6's field B** (`precision/depth_on_f5.py` `choose_field`): a bilinear G × G correction field on top of the
  plane per keyframe, Huber IRLS with second-difference smoothness, (G, λ) chosen on the held-out half A by the
  window's rule (`bend.field`).
- **MoGe-3** (`precision/moge3_runner.py`; its code and environment outside the repo, on the local disk: /root/MoGe,
  /root/envs/moge3_site_da3; weights Ruicheng/moge-3-vitg in /workspace/hf_cache): as the edge model in place of
  PointDiT (`mono_detail.runner`) or as F6's depth in place of Omega's (`bend.depth_source`).
- **No F2** (`gauge.enabled`, `precision/tie_tolerance.py`) — these two came to main.
- The experiment scripts of 2026-11-21 / 23: `arq3_vipe_run.py`, `arq3_vipe_mt_run.py` (ViPE / MetricTrail),
  `arq3_amb3r_*.py`, `arq3_publish_amb3r.py`, `arq3_select*.py`, `arq3_map*.py`, `arq3_da3_frames.py`, `v6_front.py`,
  `amb3r_launch_tests.py`.

## The numbers

Read from each session's `output/precision/refine.json` and `depth_on_f5.json` (vote shares are of the keyframes'
pixels; F5 held-out = the applied rung's held-out median).

### bufferStop

| run | F2 | F5 anchor | F6 depth | edge model | F5 held-out | coverage | contradicted | published | his eye |
|---|---|---|---|---|---|---|---|---|---|
| validated (`bufferStop`) | on | — | Omega, plane | PointDiT | 0.2549 px | 60.5 % | 15.2 % | 3,837,478 | validated 2026-11-11 |
| 1 | on | κ 49, Omega scaled per keyframe | Omega + field B | MoGe-3 | 0.281 px | 56.9 % | 19.1 % | 3,557,416 | "yo la veo bien, pero cambiemos las anclas de F5" |
| 2 | off | a scale per cube, 2 re-weights | Omega + field B (0.87 %) | MoGe-3 | 0.257 px | 56.8 % | 19.2 % | 3,572,742 | — |
| 3 | off | per cube, stable at round 5 (κ 37.2 → 8.27) | MoGe-3 (unbent 46.2 %, plane 2.94 %, field 1.10 %) | MoGe-3 | 0.256 px | 54.3 % | 22.3 % | 4,505,441 | "horrible" |
| 4 (`bufferStop_mejoras`) | off | run 3's F5 | Omega + field B | none | 0.2561 px | 58.3 % | 18.2 % | 3,745,899 | "yo la veo muy bien, tiene menos voladores" |

Runs 1–3 as reported when they ended (their sessions were overwritten by the next run); the validated and run 4 read
again on 2026-11-23. Validated / run 4 in full: kept 60.0 / 54.5 %, in place 6.2 / 6.7 %, under the support
18.5 / 23.0 %, raw points 62,126,153 / 59,849,662, the plane's held-out half B 1.11 / 1.10 %, run 4's field
(G 8, λ 0.1) 0.879 %.

### pccr 2026-08-31 (pccr 3108)

| | validated (`pccr`) | run 5 (`pccr3108_mejoras`) |
|---|---|---|
| configuration | the validated pipeline (F2 on, PointDiT on) | run 4's: F2 off, anchor (stable at round 4, κ 13.89 → 10.44), Omega + field B, no edge model |
| F5 held-out (R2 applied) | 0.5342 px | 0.5334 px |
| vote coverage | 59.9 % | 56.5 % |
| kept | 55.3 % | 51.6 % |
| in place | 6.75 % | 5.9 % |
| contradicted | 14.7 % | 18.2 % |
| under the support | 20.8 % | 23.1 % |
| occlusion veto | 5.0 % | 5.9 % |
| minority | 6.8 % | 6.5 % |
| raw → published points | 126,455,071 → 18,770,254 | 119,343,501 → 22,076,500 |
| bend held-out half B | plane 2.18 % | plane 2.16 %, field (G 8, λ 0.01 — the finest offered) 1.86 % |
| F0 → F6 | 94.4 min | 96.8 min (F4 32.8, F5 47.4, F6 14.5, check 1.1) |
| his eye | validated 2026-11-09 | "el validado sigue ganando, queda el validado" |

### ViPE / MetricTrail as the front (bufferStop, 2026-11-21 / 22)

ViPE's own pipeline (its SLAM, its metric depth fusion) and MetricTrail (with MoGe-3, 10 refinement steps, every 4th
frame and then every frame, masks off) produced the cloud in place of ours. No figure of those runs is kept here; his
words on them: *"es un desastre, sigo insistiendo que algo mal debemos estar haciendo no puede salir tan mal"* /
*"mejor pero con muchisimo movimiento"* / *"aparte hay duplicados raros"* / *"hay un tema de escala y o de posicion …
sale muy feo para ser un pipeline de nvidia"*. Then: *"utima prueba, si no funciona lo dejamos aca y por el momento
mantenemos el pipeline actual"* — and the concepts were brought into our chain instead (the runs above).

## The sessions on disk (his to keep or delete)

`bufferStop_mejoras` (run 4's cloud), `pccr3108_mejoras` (run 5's cloud).
