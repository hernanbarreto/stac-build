# STAC Build — Architecture

State of 2026-11-09. The mathematics of every reconstruction stage is in `README.md`; the stage-by-stage settings and
the reference numbers are in `docs/pipeline_final.md`. This file says how the system is built: its processes, how a job
runs, how products are stamped, resumed and published, how a session is laid out on disk, and how the UI is made.

---

## 1. Processes

| Process | Where | What it does |
|---|---|---|
| **Backend** | `server/main.py`, FastAPI + uvicorn on port 8765 (TLS), env `da3`, started by `scripts/start.sh` (tmux `backend`, log `logs/latest.log`) | REST API, the viewer / log / team / scan WebSockets, the session store, the pipeline manager. It never loads a GPU model. |
| **Stage workers** | `server/workers/*.py`, one process per stage | the work of each stage, in the stage's own conda environment (below) |
| **Precision steps** | `server/precision/*.py`, run by `precision/runner.py` | F0 → F6 inside the reconstruction stage, each step a subprocess of its own |
| **Semantic service** | vLLM serving Qwen3-VL on `127.0.0.1:8799` (`scripts/serve_semantic.sh`, env `semantic`) | the chat's model. Every VLM stage launches its own vLLM from the job's frozen configuration and stops it; the chat's service is stopped before any GPU stage and reloaded when the pipeline is idle. |
| **UI dev server** | `ui/`, Vite on port 5173 (`npm run dev:web`) | the React application |

**Environments** (`/workspace/miniforge3/envs`):

| Environment | Used by |
|---|---|
| `da3` | backend, DA3 (focal probe, I3, anchors), SAM 3.1, F0, F2, F6 and the chunk check, scale and orientation, the cleaner, the photo bake |
| `mapanything` | the VGGT-Ω fork (`vendor/VGGT-Long`), F4 (tracks), F3 (resolution probe), F5 (refinement), the sky segmenter |
| `amb3r` | AMB3R-SLAM, the whole-video first pass |
| `semantic` | vLLM |
| `genrecon` | GenRecon |
| `nodejs` | the 3D Tiles builder (`tools/glb/tile_mesh.mjs`) |

---

## 2. A job

**Orders.** `POST /api/pipeline/run {session_id, scans: [key], replace, segment | autosegment}` is the order the UI's
**Reconstruir** and **Autosegment** buttons send; `GET /api/pipelines/active` lists the running and queued jobs and their
stages. A scan key is `<date>/<source>`.

**The pipeline manager** (`server/pipeline_manager.py`) keeps one queue: one job per scan and order, one GPU job at a
time, stages in `DEFAULT_STAGE_ORDER`:

```text
RECONSTRUCTION → CLOUDCOMPY → VLM → SAM3 → PROJECTION → FLOOR → CERTIFY (off) → GENRECON (ordered only)
```

- **One configuration per job.** `config.yaml` is read once when the order is given (`read_config_once`; an error fails
  the order). The job freezes it into `output/run_config.yaml` with its per-order stage overrides, and every worker
  verifies the copy it reads. A change to `config.yaml` reaches only the orders given after it. Every parameter is
  typed; a missing key fails the load naming it.
- **Workers.** Each stage runs as a separate process with the deterministic environment (`repro.deterministic_env`:
  `PYTHONHASHSEED=0`, the cuBLAS workspace, fixed CPU threads) and the configured interpreter. It talks to the manager
  through a `WorkerPipe` (`workers/base.py`): `send_progress`, `send_log`, `send_done`, `send_error`, and
  `check_cancel` polled between steps. `run_stage_inline` runs a sub-stage (the precision core inside the
  reconstruction) as a child with the same protocol.
- **The card.** Every GPU step checks it has the whole card (`repro.require_exclusive_gpu`): a co-tenant fails the step,
  naming it; nothing is lowered to fit a shared card. The semantic service is stopped first.
- **Cancel** is cooperative with a bound (`CANCEL_ACK_BOUND_S`, about 40 s): the stage is told and may stop at a safe
  point (never inside a transaction); otherwise its whole process tree is killed. A queued job is simply dropped.
- **A backend restart drops the queue**; the orders are sent again (a stalled session with *Replace* off resumes).

---

## 3. Stamps, resume and Replace

- **Stamps** (`server/repro.py`: `stamp` / `check_stamp`, `environment_record`). Every product carries the sha256 of
  its inputs, the configuration sections it reads, its code (the files it imports, followed by
  `precision/code_closure.py`) and the environment (card, torch, CUDA). A product is reused only on an identical stamp.
- **Intake markers** (`intake/intake_state.json`): each intake step (quality, parallax, the focal probe) is skipped when
  its marker matches the frames' bytes, the parameters, the code and the environment.
- **VGGT-Ω's completion marker** (`output/maplong_run/omega_complete.json`): the fork is skipped only when the marker
  carries the stamp of exactly this launch; a marker of another launch wipes the plan's products and the fork runs.
  Inside the fork, `fork_stamp.json` resumes cube by cube; a cube or bridge of another stamp is never silently reused.
- **The precision chain** (`output/precision/chain_state.json`): the finished prefix counts while every step's input
  stamp is unchanged; the chain restarts at the first step whose stamp differs.
- **Replace OFF** resumes on all of the above. **Replace ON** empties the scan to its frames, their manifest, its
  video and its capture inputs (`intake/`, `output/` and `user_prefs/` go).

---

## 4. Geometry epochs

`server/correction/` (`apply.py`, `epoch.py`, `ledger.py`, `replay.py`, `restart.py`).

- Every correction that moves geometry publishes an **epoch**: a journaled transaction (`_tx_epoch_<N>/`) whose swap
  files the previous versions of the artifacts it replaces under `_epoch_<N-1>/` and records the transform
  (`output/corrections/epoch_<N>.npz`), the ledger (`corrections.jsonl`) and `geometry_epoch.json` (epoch, parent,
  kind). An interrupted swap rolls back. Epoch numbers are never recycled (`next_epoch`).
- F2 and F5 publish pose-only epochs; F6 publishes the cloud (kind `new_cloud`); the floor step a rigid motion.
- **After the cloud is published** only it is shown: the stores on the live epoch's lineage move to
  `output/_restart_epochs/` (out of the viewer's epoch list) and every other store is deleted
  (`restart.stash_lineage`, called by `map_worker._discard_previous_epochs` and the floor worker).
- **The way back.** When the precision chain must start again from its first step and no finished step has moved the
  geometry, the runner returns the stored lineage and selects epoch 0 (`restart.restore_epoch0`,
  `runner.needs_epoch0`) — the journaled `select_epoch`, nothing destroyed. A session with no epoch 0 kept fails there,
  naming it.

---

## 5. A session on disk

```text
server/projects/<session>/
├─ scans/<date>/src_<source>/
│   ├─ source_video.mp4           # the original video
│   ├─ inputs/                    # the rest of the capture (VIO, LiDAR, …), never deleted
│   ├─ frames/                    # the extracted frames (long side ≤ 1024) + manifest.json (sha256 of each)
│   │                             #   + selected_frames.json / witness_frames.json (the keyframes)
│   ├─ intake/                    # quality, parallax, focal probe (focal_probe.json, focal_depth.npz), AMB3R's
│   │                             #   whole-video pass (amb3r_video.npz, in metres), coverage_fill.json, walk.json,
│   │                             #   turn_frames.json, cubes.json (+ png), omega_resolution.json, their stamps
│   ├─ output/
│   │   ├─ run_config.yaml        # the job's frozen configuration (+ .sha256), run_environment.json
│   │   ├─ maplong_run/           # the fork: cube results, context views, bridge fits, chunk_graph.json,
│   │   │                         #   fork_stamp.json, omega_complete.json
│   │   ├─ omega_run/             # VGGT-Ω's per-keyframe records (epoch-0 depth)
│   │   ├─ precision/             # chain_state.json, tracks, refine, depth_on_f5.json (the vote's report), layers
│   │   ├─ cleaned_cloud.ply      # THE published cloud: x y z rgb confidence frame_global pixel_row pixel_col
│   │   ├─ potree/                # its octree
│   │   ├─ camera_poses.txt, camera.json, intrinsic.txt, geometry_epoch.json, corrections/, corrections.jsonl
│   │   ├─ _restart_epochs/       # the lineage's stores, the way back to epoch 0
│   │   ├─ vlm_analysis.json, segmentation_result.json, instance_ids.npy, scene_r.db, floor_session.json
│   │   ├─ genrecon/              # the raw GenRecon mesh
│   │   └─ tsdf/genrecon_photo/   # the published, photo-textured mesh and its 3D Tiles
│   └─ user_prefs/<user>.json     # each user's viewer preferences for this session
└─ merged/                        # project-level derivatives (merged cloud, its octree)
```

Deleting a session (`DELETE /sessions/{id}`, admin) removes the whole project folder, preferences included, and is
logged to the project audit. A point's origin (`frame_global`, `pixel_row`, `pixel_col`) is never broken by any stage.

Storage follows one rule: a session keeps what the next reader needs. The aligned chunks, the bridges' predictions
(only their fits stay) and the I3 windows are deleted once read.

---

## 6. The UI

`ui/` — React 18, TypeScript, Vite, Three.js.

- **Shell** (`ui/src/layout/`): `AppHeader` (File / View / Tools / Help menus, the session and scan pickers),
  `ActivityBar`, `SidePanel` (sessions, instances, layers), `Inspector`, `BottomDock` (console, jobs), `StatusBar`,
  `CommandPalette`, `SettingsDialog` (density, language), `SceneLookDialog`.
- **Viewer** (`ui/src/components/Viewport.tsx`, `ui/src/components/viewport/`): the cloud streamed from the Potree
  octree by `PotreeLoader.ts` with a point budget and a confidence slider (the raw support normalised by the octree's
  min and max — the arithmetic GenRecon's adapter uses); meshes streamed as 3D Tiles; measurement tools; the
  navigation cube and HUD (`ViewportHud.tsx`: gravity, scale bar, the grid's cell and major spacing, legends,
  readouts); the cubes of the cut; the segmentation instances and their oriented boxes.
- **Background and grid** (`viewport/sceneLook.ts`, View menu): per session, the background (Default — the charcoal
  gradient; Horizon — a sky dome over a translucent green ground at $y=0$; Plain — a chosen colour) and the grid (cell
  0.1–5 m, a major line every 2/4/5/10 cells, its colour). The grid's lines sit at whole cells of the world and its
  border closes on a major line.
- **Preferences** are saved per session and per user (`GET/POST /api/sessions/{id}/prefs` → `user_prefs/<user>.json`):
  point size, confidence, budget, light, opacity, background and grid.
- **Design system**: every colour is a token (`ui/src/styles/tokens.css`, read in 3-D code through
  `viewport/palette.ts`); no inline style but CSS variables; every visible string in `ui/src/i18n/` (es-MX, en) with a
  glossary of terms. `npm run verify:design` runs `scripts/check-tokens.mjs`, `check-contrast.mjs`, `check-inline.mjs`,
  `check-strings.mjs`, the linter, the type check and a build.
- **Other panels**: the spatial assistant (chat; the VLM proposes, deterministic tools measure), BIM / IFC registration
  and deviation, team workspaces, the WebXR view.

---

## 7. Vendored code

| Directory | What | Branch / pin |
|---|---|---|
| `vendor/VGGT-Long` | the VGGT-Ω cube run, seams, revisit bridges, the Sim(3) graph, the decision rule | fork, branch `gpu-main` |
| `vendor/AMB3R-SLAM` | the whole-video first pass and its light map | fork, branch `stac` (third party: never pushed upstream) |
| `vendor/depth-anything-3` | DA3 | pinned |
| `vendor/pointdit` | PointDiT-H/16 (edge detail) | pinned, sha256-verified weights |
| `vendor/GenRecon` | the scene mesh | pinned commit + `server/tools/genrecon/genrecon_stac.patch` |
| `vendor/PotreeConverter` | the octree, with the canonical chunk order | patched |
| `vendor/mvs-texturing` | texrecon, kept | — |
