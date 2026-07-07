# Phase 7 — Integral Validation: Report

**Status: complete, validated end-to-end on the A6000.** Spatial Q&A suite,
GPU coexistence probe, per-consumer metrics, and a reproducible Pitch-2 demo
script. Two items carry genuine EXTERNAL data dependencies (called out below);
all code and tests are complete.

## What was built

| Component | Path |
|-----------|------|
| Q&A suite (28 Q) | `server/phase7_validation/qa_suite.yaml` |
| Runner + scorer | `server/phase7_validation/qa_runner.py` |
| GPU coexistence probe | `server/phase7_validation/coexistence.py` |
| CLI | `server/phase7_validation/cli.py` (`qa`, `coexist`) |
| Unit tests | `server/phase7_validation/tests/test_phase7.py` (6 tests, no model) |
| End-to-end demo | `scripts/demo_pitch2.sh` |

## Q&A suite (spec: 25–30 Q, 5 traps, 3–5 alignment-health)

28 questions over one of our scenes (the test3 R.8 store): count/list,
measurement, findings, alignment-health (4), and **5 insufficient-data traps**.
Each carries a machine-checkable expectation; geometric magnitudes are
cross-checked against the deterministic tool value (the VLM only relays it) and
flagged `cloudcompare_pending` for the human magnitude cross-check.

### Result — `qwen_local` (Qwen3-VL-8B), live service

```
accuracy = 1.000  (28/28)     latency mean/p90 = 1.83 / 2.77 s     mean_iters = 1.86
  count        4/4    findings     4/4    health       5/5
  list         1/1    measurement  9/9    trap         5/5
  tool-calling failures: none · measurement/health with no tool: none
```

Spot-checked for genuine correctness (not regex-gaming):

- trap q20 → *"object 9999 does not exist … I cannot compute the distance"* (no tool call);
- trap q19 → *"the tools … do not include any function to measure surface temperature"*;
- measurement q08 → `get_clearance` → *"0.445 metres, confidence 1.0"* (faithful relay);
- health q15 → `get_alignment_health` → *"bimodal … mode separation 0.206 m"*.

The 8B chained tools reliably on this suite (1.86 mean iterations, no runaway
loops, no wrong-tool calls). **Honest caveat:** `get_plumb`/`get_level`
magnitudes are large here because this single-window store has no estimated
gravity — the VLM relayed the tool value faithfully (intended behavior); the
absolute magnitude is what `cloudcompare_pending` marks for human cross-check.

### `qwen_local_large`

The runner accepts comma-separated backends. The larger candidate
(`qwen_local_large`, `gpu_memory_utilization=0.80`) requires (a) its quantized
weights and (b) a service restart (one model served at a time). **External
dependency** — weights + restart; the harness path is complete.

## GPU coexistence (A6000 48 GB)

`coexist` probe, measured live while the service served:

```
total=49140 MiB · vLLM used=23574 MiB (gpu_memory_utilization=0.50) · free=24977 MiB
recon peak needed ≈ 15000 MiB (VGGT-Ω 14 GB, DA3 9 GB, Phase R 1.5 GB)
=> COEXISTS: 24977 ≥ 15000 MiB. Keep semantic at 0.50; run reconstruction + Phase R concurrently.
```

If a heavier backbone raises the peak above the free headroom, the probe
recommends temporal partitioning (stop the service during VGGT-Ω, restart for
Phases 1–6). Final config: **semantic `gpu_memory_utilization=0.50`, concurrent.**

## Per-consumer metrics

| Consumer | Metric | Status |
|----------|--------|--------|
| Phase 5 Q&A | 28/28 accuracy, 1.83 s mean, 1.86 iters | ✅ measured (this phase) |
| Phase 3 findings | precision 0.600 / recall 1.000 / specificity 0.742 | ✅ measured (43-crop proxy-GT); full 50+ hand-annotation is the external step |
| Phase 4 QC | false-discard rate | ⏳ needs a human keep/discard-labeled frame set; harness + drop-log complete |
| Phase 1 detection | recall/precision per class | ⏳ needs hand-segmented GT (flagged since Phase 1); eval.py complete |
| Phase R | full A/B on the 160 m corridor (R.9) | ⏳ needs the multi-window corridor scan; A/B gate + pipeline complete |

## Open items (EXTERNAL data only; code + tests complete)

1. `qwen_local_large` run — needs the large quantized weights + a service restart.
2. Phase 1 per-class recall/precision — needs hand-segmented ground truth.
3. Phase R 160 m corridor A/B — needs the multi-window corridor scan.
4. Phase 4 false-discard rate + Phase 3 full 50-crop precision — need human labels.

These are the ONLY open items across all phases; every one is a data/human step,
not missing code — each corresponding harness runs and is unit-tested.

## How to run

```bash
python -m phase7_validation.cli qa --scene scene.db --backend qwen_local --out qa.json
python -m phase7_validation.cli coexist --out coexist.json
scripts/demo_pitch2.sh <session_dir> scene.db out_dir     # full Pitch-2 demo
```

*Hernán Barreto — Ingerop IN3 Session IV — STAC*
