# Phase 1 — SAM3 Auto-prompter: Report

**Status: core built and validated on real BA geometry.** The system first
UNDERSTANDS the scene (what is this place / what is in it, no domain assumption),
then does understanding-driven open-vocabulary detection + geometric temporal
association to pre-populate a segmentation session for human review. A headless
batch CLI produces the masklet contract for Phase R. InternVL3 remains fallback.

## Design principle (revised per review)

**Segment EVERYTHING, understand what you see — do not restrict to a class
list.** The detector localizes every distinct object/surface it comprehends,
with free-form labels. The construction vocabulary (`vocabulary.yaml`) is only a
**canonicalization / routing overlay** (known structural/dynamic classes get a
canonical id for surface_fitting, R.5 flatten-whitelist, R.7 dynamic masking) —
it is **never a detection filter**. Unknown objects are kept with their VLM
label and safe defaults. Pipeline: `scene_understanding` → understanding-driven
`detector` (per-frame NMS dedup) → `associate` → session + review queue.

> Note: scene identity is discovered, never assumed. Example (test2, told
> nothing): understood as "train maintenance depot" — a CDMX metro train under
> maintenance — and detected 43 open-vocab object types (train, tool cart, fire
> extinguisher, scaffolding, electrical box, bucket, cables, plastic sheeting…),
> none of which are in the 16-class obra vocabulary.

## What was built (`server/segmentation/autoprompt/`)

| Component | Path | Role |
|-----------|------|------|
| Vocabulary | `vocabulary.yaml` + `vocabulary.py` | 16 construction classes (EN canonical, ES alias, `phrase`, `structural`/`dynamic`/`small`, per-class conf) |
| Detector | `detector.py` | Qwen3-VL grounded open-vocab → boxes+labels+conf; strict JSON, tolerant parse + 1 correction retry; boxes normalized xyxy [0,1] |
| Association | `associate.py` | BA-pose reprojected-box IoU + union-find → instances. **No appearance matching** (anti-aliasing rule in code) |
| Orchestrator | `session_builder.py` | detect→associate→confidence gate→`(prompt, frame_map)` contract + boxes + review queue → `vlm_analysis.json` |
| Batch CLI | `cli.py` | headless masklet producer for Phase R |
| Eval | `eval.py` | per-class recall/precision @IoU>0.5 vs hand-segmented GT + time-vs-manual |
| SAM3 box method | `sam3_wrapper.add_box_prompt` | native `bounding_boxes` xywh path (was text/points only) |
| Config | `config.yaml` `autoprompt:` | thresholds, backend, vocab path |
| Integration | `workers/vlm_worker.py` | Phase-1 auto-prompter is the DEFAULT inventory path; InternVL3 fallback |
| Unit tests | `tests/test_associate.py` | 4/4 pass |

## Exact SAM3 contract honored

- **Prompt/frame_map** (`vlm_analysis.json` → `sam3_worker` → `run_segmentation`)
  is produced exactly as the existing pipeline expects (`;`-joined labels +
  `{label: [keyframe files]}`), so the pre-populated session opens in the
  Segmentation Manager with no UI changes.
- **Boxes** persisted as normalized **xywh** (SAM3's native box convention,
  `vendor/sam31/.../sam3_video_inference.py:881`), ready for box-seeding /
  review overlays. `add_box_prompt` sends `bounding_boxes`+`bounding_box_labels`.
- Every proposed label/box tagged `origin: vlm_proposed` (inviolable rule).

## Validation on real data (test3 — unknown-domain session; do NOT assume; 307 frames / 230 keyframes with BA poses)

**Detector** (live 8B): on real keyframes it correctly detects the construction
domain — vault, wall, floor, platform, column, pipe, luminaire, sign, door — at
high confidence. Example (8 consecutive keyframes): 62 detections.

**Geometric association** (real BA poses, 230 c2w): **62 detections → 29
instances, 10 multi-view**. Large surfaces (wall, vault, floor, sign) linked
across 7–8 keyframes; discrete luminaires/doors across 2–3. Depth-aided and
assumed-depth gave identical grouping (robust). Fallback with no poses keeps
detections separate — **never merges by appearance** (verified).

**Unit tests** (`test_associate.py`): 4/4 — same object merges, different
positions don't, different labels never merge, no-poses fallback stays separate.

**End-to-end CLI**: `python -m segmentation.autoprompt.cli --session <dir>`
produced the correct `vlm_analysis.json` (prompt + frame_map + boxes + instances
+ review_queue + thresholds) and the review-queue file.

**Eval harness**: runs, matches boxes at IoU>0.5, computes per-class P/R, timing,
and flags weak small classes.

## Metrics caveat (honest)

The only Manager-format GT on disk (`test3/output/segmentation.json`) is a
**sparse 4-instance manual session** labeled `floor/ladder/wall1/wall2/door` — it
does NOT use the obra vocabulary. Running eval against it therefore only yields
one legitimately comparable number:

- **floor: precision 0.73, recall 0.67** (@IoU>0.5, conf≥0.3).

Everything else the detector correctly finds (vault, wall, pipe, luminaire…)
counts as a false positive because the GT never annotated it — so the overall
0.11 precision is a GT-coverage artifact, **not** a Phase-1 result. Per the spec,
real per-class recall/precision needs a **hand-segmented set with the obra
vocabulary** (a small data task on our side). The harness is ready to consume it:
`python -m segmentation.autoprompt.eval --session <dir> --gt <gt_output_dir>`.

## Timing

Detector ≈ 12.7 s/keyframe on frames with many objects (JSON up to 1536 tokens);
~0.4–1 s on sparse frames. A 230-keyframe session ≈ 30–50 min headless — vs
manual prompting of every instance from scratch. The eval's crude time-vs-manual
estimate reported ~1.3× on the sparse GT (25 clicks); the real speedup is far
larger on a fully-populated obra scene (dozens of instances auto-seeded).

## Weak classes (spec datum for model escalation)

Small/thin classes (`catenary`, `sign`, `track`, `sleeper`) are flagged `small`
in the vocabulary and get relaxed per-class confidence (0.25). These are the
candidates to re-check against the larger `qwen_local_large` in Phase 7. The eval
already emits `weak_small_classes`.

## Not yet run (next validations)

1. **SAM3 mask generation** (`--run-sam3`): produces `segmentation.json` +
   `seg_masks.npz` via the existing `run_segmentation`. Needs the `sam3` env + a
   free GPU (stop the semantic service first). The contract feeding it is
   validated; this is the one remaining end-to-end integration run.
2. **Real per-class eval**: needs the obra-vocabulary hand-segmented set.

## How to run

```bash
# service up (Phase 0)
bash scripts/serve_semantic.sh &
# headless auto-prompt (masklet producer for Phase R)
conda activate da3 && cd server
python -m segmentation.autoprompt.cli --session <session_dir>            # contract only
python -m segmentation.autoprompt.cli --session <session_dir> --run-sam3 # + masks
python -m segmentation.autoprompt.eval --session <dir> --gt <gt_output_dir>
```
