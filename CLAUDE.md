# CLAUDE.md — working rules for this repo

Compacted 2026-11-08 at the user's request ("hay mucha basura y cosas obsoletas"). The full chronological record —
every decision with its date, his words and the measurements behind it, 2026-08 … 2026-11-08 — is in git:
`git show 2f13136:CLAUDE.md`. The pipeline stage by stage, its numbers and its pending lists: `docs/pipeline_final.md`.

## 1 · ⭐⭐⭐⭐ THE FINAL PIPELINE — USER 2026-11-08 (11:20 CDMX), pccr, main 8cef74f
*"la cobertura es muy buena, sin errores, sin duplicados, muy muy buen resultado, excelente, que quede este como
pipeline final, dejamos terminar todas las sesiones, todas"*. DO NOT TOUCH IT WITHOUT HIS WORD.
- intake I0 / I1 / I3 → AMB3R-SLAM over the whole video at stride 1 (*"amber solo decide las poses, los cubos como se
  arman lo decidimos nosotros"*) → coverage fill (one second view ≥ 2°) → turn frames → cubes.py 9 (≤ 5 m and ≤ 64
  views along the walk, neighbours sharing half their keyframes; `kf_covisibility` c > 0: *"ven lo mismo es ver
  total o parcialmente el sector"*) → Omega per cube + ONE Sim(3) graph on the seams alone → F0 F2 F4 F3 F5 → F6:
  Omega bent to F5's landmarks + PointDiT at the edges + the vote of EVERY co-visible keyframe (widest baseline
  first), a pixel published only with ≥ 10 views ≥ 2° that test its depth (*"prefiero poco y bueno que mucho y
  mierda"*) → cloud → [VLM → SAM3 → rounds → projection] → floor (automatic). GenRecon manual; certification OFF.
- pccr: coverage 63.6 %, 11.46 M points, F5 held-out 0.462 px, 491 segments (`docs/pipeline_final.md` §2).
- The queue of 2026-11-08 (his order, Replace + segmentation): pccr ✓, observatorio, zaragoza, bufferStop, station,
  pccr 2026-08-24 ("2408"); then GenRecon of all six (*"para tener todas las nubes y todos los genrecon"*) at
  confidence 0 % and ONE point per active voxel (`genrecon.min_points_per_voxel: 1`, USER 2026-11-08, commit 475ac0c
  *"ok, dale, ajustalo así, que quede"* — at 5 thin pipes and sparse wall were eaten; the queue's jobs carry it).
  Closing the gaps no keyframe saw through: later, on his word.
- GenRecon grid (USER 2026-11-08 ~04:00 CDMX, *"grilla anclada en y = 0 … el suelo es sólido … nada más cambia"* /
  *"dale, si, no te detengas hasta que quede implementado"*): one cube base EXACTLY on the lowest measured support
  level (`anchored_centers`; layers down only where the cloud reaches lower — full coverage of 2026-11-05 kept) and
  every support level's active columns filled SOLID to the lattice base (`plan_layout`) — against the double floor
  (both faces of the floor's voxel slab) and the cabinet-base detail lost into the floor's layer. ONE z-centers
  source for levels.json + layout.json. Verification on the queue's GenRecon runs: the mesh height histogram must
  give ONE floor plane (the 08:44 pccr mesh, min 5 pts, gave two) + cloud→mesh distance at the cabinet base.
- bufferStop (a −90° video) comes out of the reconstruction ON ITS SIDE; the floor step rotated it 92.74° on the
  segmented floor (1.03 M pts under all 315 cameras, 0.99 m below) — USER 2026-11-08: *"bufferstop está bien
  apoyada"*. The orientation step's miss on rotated videos stays declared, not investigated.
- He presents the pipeline at Ingerop's IN3 contest about a month after 2026-11-08 and believes Ingerop may own it
  (*"antes tal vez no me dejen porque es de ellos"*). Nothing goes public (arXiv, open code, outside talks) without
  his explicit order.

## 2 · HOW TO WORK WITH HIM (hard-earned — each line cost him time or money)
- NO DECISION WITHOUT HIS OK (2026-10-04): not a config default, not a UI default, not a repair on his session — one
  line, wait for the OK. Carrying out an order he gave is not a decision; widening it is.
- "detené todo" / "detenete": stop every tool call NOW, kill what he names, confirm with the process list, answer in
  one line, wait. No new launch without his word.
- His EYE is the verdict; my metrics inform, never decide (*"tus métricas son incorrectas"*, 2026-11-04). Never claim
  a material or a cause without looking (*"si te digo eso es porque sé lo que hay"*). Never report a cause or "it is
  fine" before a FINISHED command shows it — the empty output of a backgrounded command is no evidence (2026-11-08).
- THE POD COSTS MONEY BY THE MINUTE (2026-11-08): a stalled run gets two quick checks (the STAGE process's CPU and
  the card), then the fix (§4); never a long investigation while the card idles.
- ONE task at a time; one GPU job at a time (A/B timing under contention is invalid); never several agents on his
  session; never change a validated recipe on a hypothesis; when porting, reproduce its numbers EXACTLY on the real
  session before saying it is done.
- Long runs: a performance checkpoint in the first minutes; progress reported unprompted (~20 min — silence reads
  as a hang); a stall detector that reads the stage process and the card, not the log alone (F5's rungs are silent
  4–6 min at 100 %).
- GPU pipeline runs: he launches them, or I do on his explicit order. Tests, agents and workflows are allowed within
  the pod's limits (§4) (2026-09-28).
- A task is closed when implemented, wired, tested and committed — never leave a pending item silently. Record his
  verdicts here with the date and his words. Short answers; no tests nobody asked for.
- He writes Spanish (rioplatense); everything in the repo — code, YAML, comments, docstrings, commits — is ENGLISH.
- Commits on main, one per block, his words in the message; GPU ports on `gpu-main`, fast-forwarded after the suite.
  Nothing is pushed without his order; the git history is never rewritten (2026-11-08: *"dejalo así"*).

## 3 · ENGINEERING RULES
- PROVENANCE (inviolable): the VLM proposes / describes / detects / classifies / orchestrates — it NEVER measures.
  Every metric comes from deterministic tools over geometry; every VLM output entering a deliverable is tagged
  `vlm_proposed` / `tool_measured` / `human_validated`.
- SEGMENT EVERYTHING: the auto-prompter understands the scene with no domain assumption; the construction vocabulary
  is a canonicalisation / routing overlay, never a detection filter.
- NO INVENTED NUMBERS (*"criterio unificado y correcto, sin números inventados"*, 2026-09-25): a value that gates is
  measured or his. The bar is the sample's own noise — `metric_lock.decide_change` / `heldout_change`: 95 % CI on one
  side, ≥ 5 judges, ≥ `improvement_error_factor` (1.1, *"1.1 en todos los casos"*) × the measured error. Gates are
  ADVISORY: measured, declared, the correction applied (`veto` for evaluation only). A gating number that misbehaves
  is replaced by a measurement, never retuned. His numbers in the final pipeline: cube 5 m, ≤ 64 views, half
  overlap, c > 0, ≥ 10 views ≥ 2°, Omega ≤ 1024 (*"la resolución de Zaragoza siempre 1024, no más"*), factor 1.1, a
  visit = 1 m of walk (`correction.visit_drift.min_walk_m`), the one confidence floor of 10 % (`conf_min_norm`).
- CONFIG: every parameter in `config.yaml`, typed; a missing key fails the load naming it, a leftover of a removed
  key fails too; an exposed vendor default reproduces the vendor's behaviour; vendor keys are written in the
  BUILDERS (`_apply_stac_model_keys`), never at a call site — three silent wiring bugs came from that (2026-09-23).
- DETERMINISM (2026-10-07, *"STAC Build es una herramienta de ingeniería, debe dar siempre lo mismo"*): bit-for-bit,
  `docs/plan_determinismo.md`; strict deterministic torch, exclusive GPU, every reused product on its stamp, no clock
  in a compared artifact.
- GPU PORTS (2026-11-01, *"todo debe correr en GPU, en modo determinista … la GPU no debe quedar parada nunca"*):
  measure first on the real session (cProfile / a timed log; pccr read-only or a copy under /workspace/tmp); the
  torch twin in float64 on ONE declared device with the numpy function KEPT as the path of record; a test holding
  both equal (exact for integer decisions and verdicts, 1e-9 for geometry); the device key typed (`numpy | cpu |
  cuda`; cuda with no card FAILS naming the key); tests on cpu (`tests/conftest.py use_device("cpu")`). Stays on the
  CPU, declared: PotreeConverter (one thread — byte identity), sqlite, PLY / JSON / npz I/O, JPEG decoding.
- ARCHITECTURE: the server process never loads a GPU model — workers in subprocesses (`WorkerPipe`: progress, log,
  done, error; `check_cancel`; free the card on exit); paths only through `project_paths.py`; a point's origin
  (`frame_global`, `pixel_row`, `pixel_col`) is never broken; Potree LOD through `PotreeLoader`; geometry epochs are
  transactional and every derived artifact carries its epoch.
- Read before writing; trace the whole path (worker → pipeline manager → API → UI); never swallow an exception, hide
  a failure or leave a workaround that masks a bug — fix the root cause.
- Never edit a bash script in place while it runs (bash reads it by offset): write a new file and `mv` it over.

## 4 · THE POD
- NVIDIA A100 80 GB PCIe (sm_80), driver 595, torch cu128. cgroup v1 limits — not `free` / `nproc`: 117 GB RAM,
  ~30 CPUs; over 117 GB the whole container dies. Baseline (backend + vLLM) ≈ 25 GB; cap heavy jobs `(ulimit -v
  30000000; OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENCV_NUM_THREADS=2 timeout <s> taskset -c
  0-7 nice -n 15 …)`; never a GPU bench under `ulimit -v` (CUDA maps device memory: a false OOM).
  `memory.usage_in_bytes` counts the page cache — read `memory.stat total_rss`.
- /workspace is RunPod's MooseFS FUSE volume (`mfs#ca-mtl-3.runpod.net`), quota 400 GB invisible to `df`: a full
  quota fails every write with Errno 122 and the worker dies without a traceback. The local disk (/root/.cache: torch
  hub, the VGGSfM tracker, PointDiT's DINOv3 3.1 GB copy) is wiped by a pod restart and refilled once.
- INCIDENT 2026-11-08 (an hour of his pod): at 09:12 CDMX everything stopped during F5 — no traceback, no OOM, the
  log silent until the pod restarted at 10:14; cause not determined (a provider stop or the volume freezing). The
  first resume then hung in `intake.run`: its main thread in the kernel's `request_wait_answer` (a FUSE request never
  answered), 0 % CPU, nothing written. THE FIX: kill the stuck worker, RESTART THE BACKEND (its job record stays
  "running" and refuses every order), relaunch with replace OFF (it resumed at F5, R0–R2 bit-identical). A backend
  restart DROPS THE QUEUE: re-send the stalled session with replace OFF and the rest as they were, in order.
- Services: backend `bash scripts/start.sh` (tmux `backend`, port 8765, `logs/latest.log`); semantic service `bash
  scripts/serve_semantic.sh` (vLLM / Qwen3-VL on 127.0.0.1:8799). Every VLM stage launches its OWN vLLM from the job's
  frozen config and stops it; the chat reloads it when the pipeline is idle. From a shell vLLM needs `setsid nohup`
  (launched from a short python it dies with it) and ≥ 40 GB free. Restart the backend to pick up new stages / routes.
- API: `POST /api/pipeline/run {session_id, scans: [key], replace, segment: [key]}` (the Reconstruir button);
  `GET /api/pipelines/active` (running / queued jobs and their stages); scan keys `<date>/<source>` from `GET
  /sessions/<id>/scans`. A session lives in `server/projects/<id>/scans/<date>/src_<source>/`.
- Storage (2026-10-05, *"que no almacene al pedo"*): a session keeps what the next reader needs
  (`certify.keep_aligned_chunks: false`, bridges without images, I3 windows deleted after the chain). Replace ON wipes
  to frames + video; OFF resumes on stamps. Quota 450 GB (2026-11-08: ~358 used before pccr 2408; a ~2000-keyframe
  session peaks at ~70–95 GB). Safe to free: the pip cache, `conda clean --tarballs --packages` (no env softlinks
  into pkgs), `vendor/GenRecon/work` (every GenRecon run rmtree's and recreates its own dirs) — USER 2026-11-08
  "ok, sin romper nada" / "los genrecon que se usaron a modo de pruebas eliminalos" (37 GB freed).
- Envs (`/workspace/miniforge3/envs`): da3 (backend + core + PointDiT), mapanything (Omega, F4, F5), semantic (vLLM),
  amb3r (AMB3R-SLAM), genrecon, CloudComPy310, gs, nodejs, colmap (evaluation only). One HF cache: `HF_HOME=
  /workspace/hf_cache`. No paid external APIs; everything local.
- texrecon (`vendor/mvs-texturing`, kept *"por las dudas"*) was relinked against libtiff 6 (its own link.txt,
  `libtbb.so.12.19` → `libtbb.so.12`); a full `cmake ..` fails on the vendored oneTBB.

## 5 · THE APP AROUND THE PIPELINE
- Reconstruir: per-scan *Segment when done* (OFF by default) and Replace (OFF = resume). Instances → Autosegment:
  the VLM prompt + SAM3 prompts (editable, saved in `output/autosegment.json`) and a checkbox per stage, GenRecon
  with the viewer's confidence. Instances → Cubes (each cube as its own box). The Fuse icon sits on the PROJECT row
  of the Sessions tab when it holds two or more scans.
- Chat = spatial intelligence (`phase5_qa`): runs in a worker thread (never on the event loop); tools for measuring
  (feature bands, vertical axis, extents), `describe_scene` (VLM, cached), persistent notes, evaluation VOLUMES
  (rest on the floor, gizmo Move / Rotate / Resize, collision tint, `measure_volume`); a session with no instance
  store gets it rebuilt from `segmentation_result.json`.
- Segment cleaning: RANSAC on, DBSCAN OFF (*"este sí borra puntos"*), the 5 cm voxel quads OFF; OBBs fitted to the
  points. Viewer: segmentation lamps / windows as lights, point opacity, shadows for meshes.

## 6 · TRIED AND DISCARDED — do not bring back without his word (details in `git show 2f13136:CLAUDE.md`)
- AS THE RECONSTRUCTION: VGGT-SLAM 2.0, VidMap, AMB3R's own cloud (2026-11-04: none beat the pipeline); LoGeR +
  PointDiT + DA3 (2026-10-06); the DA3-streaming cloud (*"no es mejor ni cerca"*); DA3 depth in the cloud (*"capas
  duplicadas y puntos voladores"*); the plane sweep / witness fusion (`cloud.source: fusion`, selectable).
- THE CUT: one pass for the whole walk; fixed 60/30 chunks; walk-sized 12 / 15 m chunks; chunks planned by
  co-visibility lengths (2026-09-22 … 10-06); cubes grouped by what the keyframes see (cubes.py 3–6); 5 m cubes with
  no view cap (2026-11-07 night: bent edges, a doubled floor, the corridor's door lost, 6 h 28 min).
- THE GRAPH: DA3 per-cube priors and Cauchy on the seams (broke seams by 16–33 cm); SALAD on cubes (revisits come
  from AMB3R's map); the "3× drift budget" bar; the frustum veto of SALAD pairs; a top-view silhouette witness.
- F6: witnesses capped at 18; the NEAREST co-visible witnesses (20 cm apart: the corridor's door lost); τ 0.3 of
  co-visibility; the coverage fill to 10 views (686 keyframes: more cubes, more time, worse); PointDiT detail on every
  surface (bent walls); the "≥ 3 views within τ_rel" confirmation; the "witness ≥ 1 m of walk" rule.
- MESHING (removed 2026-10-16/17 with their code and envs): PGSR, TSDF, Poisson, NKSR, ShapeR, MeshFlow, forms,
  hull, plane segments, Manhattan snapping, point2cad (CC-BY-NC); GenRecon's generative variant (*"inventó cualquier
  cosa"*), `cloud+unseen` (*"lo que alucina lo alucina mal"*), `cloud+objects` (duplicated objects). COLMAP + OpenMVS
  on pccr (2026-10-17, `docs/openmvs_vs_genrecon_pccr.md`): COLMAP's SfM drifts 12.3 cm rms on this video; OpenMVS's
  mesh is crumpled paper — a distance-to-cloud score rewards a mesh that hugs the noise.
- CORRECTION: the certification (OFF 2026-11-05, *"la certificación de mierda esa"*); the greedy loop (2026-09-18);
  the manual Corrections button (2026-09-24); the chunk gizmo (2026-09-08); the per-chunk floor model (bufferStop
  *"de costado"*, 2026-10-16). RULED OUT BY MEASUREMENT (2026-09-19, pccr): a global rigid rotation, per-object
  rotation, a uniform scale of a copy, camera rotation — the duplicates lay ALONG the line of sight (depth).
- SEGMENTATION: the VLM on a sample of keyframes (it saw 8 of 289); 2×2 crops; one prompt per head noun (dropped 22
  of 52 phrases); a second VLM pass over the unsegmented points (2026-10-04 — the rounds, which grey what is
  segmented, replaced it).

## 7 · PENDING
- The pipeline's and the general lists: `docs/pipeline_final.md` §4 — capture distance (when he says), the weighted
  vote (his word), smarter segmentation, epoch renumbering, the GPU ports of the intake, MEJORAS_OBLIGATORIAS
  (never built), the brush lock, the BIM comparison, viewer checks.
- An evaluation against laser ground truth (ScanNet++ / ETH3D) for the IN3 and a later paper: proposed 2026-11-08,
  not ordered.
- `docs/`: ARCHITECTURE.md and ROADMAP.md are out of date — update or delete on his word. His own documents
  (SCANNING_GUIDE, FUTURE_VISION, COVERAGE_STRATEGY, BIM_INTEGRATION_PLAN, the ADRs 001 / 002, FIREFOX_CONFIG, the
  phase reports signed "Ingerop IN3 Session IV") are his to keep or drop.
