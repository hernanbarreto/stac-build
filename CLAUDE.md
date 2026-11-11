# CLAUDE.md — working rules for this repo

Compacted 2026-11-08 at the user's request ("hay mucha basura y cosas obsoletas"). The full chronological record —
every decision with its date, his words and the measurements behind it, 2026-08 … 2026-11-08 — is in git:
`git show 2f13136:CLAUDE.md`. The pipeline stage by stage, its numbers and its pending lists: `docs/pipeline_final.md`.

## 0 · ⭐⭐⭐⭐⭐ THE NEW CANDIDATE PIPELINE — USER 2026-11-08 (evening CDMX), pccr 3108, main 14a731a
*"en pccr que acabo de ver está perfecto, sin duplicados, sin voladores, con excelente cobertura, sin agujeros, es decir
todo OK, felicitaciones, anotalo, y si soporta el resto de las sesiones es el pipeline"* — and: *"ninguna nube de las
que he visto reúne todos los requisitos de perfección y completitud sin voladores que esta reúne"*: THE BEST CLOUD OF
THE PROJECT TO DATE, the reference every later run is judged against. Pending: zaragoza, pccr 2408, station (queued,
cloud only, replace). What it is, over §1's pipeline:
- intake: AMB3R's completeness — `parallax.max_window_frames: 8`, fixed blocks of 8 video frames + densify, never a gap
  over 8 frames without a keyframe (`intake.parallax.stride_fill`, 31e0c52); every keyframe gives birth.
- cubes: covis budget, cap 64, min 32 kf, 50 % overlap (5b6308e); the REVISITS as SLAM closures of the cube Sim(3)
  graph — F4's own pairs, one joint Omega bridge each, Cauchy loop edges (a3c435a); scale bound = the session's own
  (1.1 × the 95 % interval of its seam scales, no fixed 0.6/1.6), bridge window = F4's (ec02dc7, 14a731a).
- F6: COLMAP vote + repair to the witnesses' median (≤ 10 m) + Merrell occlusion veto, ON (b74b5e6, 6ecbe88, a51f3c7).
- cleaner: voxel 1 mm, SOR σ 6 (4287313). Replace empties output/ entirely; floor stage off (a3c435a).
- pccr 3108 numbers: 547 kf, 17 cubes, seams 1.0–3.7 cm, 43/49 closures (graph moved cubes ≤ 61 cm), F5 held-out
  0.534 px, witnesses bound 2.26 px, vote coverage 74.1 %, 132.5 M points published.
- ✓ ZARAGOZA VALIDATED (USER 2026-11-09) under the 2026-11-09 rules (no repair, majority vote, THE RANGE 15 m, AMB3R
  scaled to metres with DA3, frames ≤ 1024): *"si bien tiene poca cobertura la que tiene es excelente y lo que falta es
  todo lo que se tomaba de lejos … el tren que es el objetivo del escaneo está perfecto, el resto que es mucho de fondo
  y a los costados no está escaneado … está perfecto para mí"*. A low coverage there is the far background (beyond the
  range or seen by too few views), NOT a defect — zaragoza is walked like observatorio (a train in a shed). Numbers:
  seams 2.5–16.5 cm, graph median 3.5 / max 11.2 cm, F5 held-out 0.398 px, vote coverage 30.1 %, 42.6 M points.
- ✓ PCCR 3108 VALIDATED under the same rules (USER 2026-11-09): *"también está muy bien, con algo menos de cobertura
  pero con buena solidez en los objetos, lo que falta o está ahuecado es lo que está más allá de los 15 m, que no es
  parte del fondo del escaneo"*. Numbers: seams 1.0–3.7 cm, 43 closures, F5 held-out 0.534 px, witnesses bound 2.26 px,
  vote coverage 49.3 % (no repair), 88.8 M points; AMB3R metre 2.6381 m/unit (MAD 6.3 %).
- ✓ ZARAGOZA VALIDATED AGAIN (USER 2026-11-09 evening: "salió bien, lo único la columna partida") under the same
  three changes (5 mm / σ 3, 1.5°, F6 in place): vote coverage 42.2 % (was 30.1 %; kept 36.4 %, in place 11.1 %),
  69.5 M raw → 23.3 M points (was 10.1 M), F5 held-out 0.398 px bit-identical, core 51.6 min. Known, not
  investigated: the split column at the start of the shed (he waved it off earlier: "no importa la columna esa").
- ✓ PCCR 3108 VALIDATED AGAIN (USER 2026-11-09 evening: "bien, pccr validado visualmente, ok") with the
  cleaner back to voxel 5 mm / SOR σ 3 (4b346ae), the vote's minimum angle 1.5° (b893e5f) and F6 IN PLACE
  (bb7a871: a pixel whose witnesses agree with its own depth is published where it is, never moved): vote
  coverage 59.9 % (kept 55.3 %, in place 6.7 %), 126.5 M raw → 18.8 M points published, core 94.4 min. The
  near surfaces the no-repair run lost came back in part (hollow 1 m zones 28 → 57 % filled); what still
  stays out is held by the occlusion veto, missing votes and the majority, in that order (census).
- ✓ PCCR 2026-08-24 ("2408") VALIDATED (USER 2026-11-11), 2765 keyframes, the longest walk of the project: *"la
  nube de pcr2408 salió espectacular confirmada además por la persona que grabó el video, la geometría es correcta
  increíble! Algunos pequeños duplicados pero nada grave y en un recorrido que hasta los modelos de frontera temen,
  el nuestro logró coherencia absoluta"*. The geometry confirmed by the person who filmed it.
- ✓ BUFFERSTOP VALIDATED (USER 2026-11-11): *"no estaba acostado estaba bien parado y salió excelente también"* —
  under this pipeline it comes out STANDING (the 11-08 run's "on its side" below no longer holds).
- STATION (USER 2026-11-11): *"había salido bien, bastante bien con como se grabó … lo que estaba reconstruido
  estaba muy muy bien, gran coherencia"* — DELETED by him: many parts missing from a poor recording, not a
  session to show in December. Observatorio: not yet seen by him.

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
  Closing the gaps no keyframe saw through: later, on his word. Station's cloud ✓ 2026-11-08 ~05:00 CDMX
  (*"se ve bien, no mucha cobertura pero de buena calidad"*): coverage 41.0 %, 28.95 M pts, held-out 0.598 px
  (R2), witness bound 37.04 px — the session's own held-out tail, vs 2.96 px of the 01/02-11 station.
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
- ⭐ FRONT SWAPPED BACK (USER 2026-11-08 ~06:00 CDMX): station under the 11-08 front (AMB3R + 15 cubes ≤ 64 views)
  gave NON-RIGID seams of 60/32/20 cm (the same pixels reconstructed apart in neighbour cubes — no graph closes
  that), a −7 % Omega scale ramp along the walk, a split circular column and a doubled wall (*"esta corrida salió
  con poca cobertura y con duplicados, es una bosta"*); the behind-copy survives F6's vote because it is HIDDEN,
  not seen through (his 2026-09-19 finding). The good station was 01/02-11 (main 87d7c00): co-visibility chunks of
  67–127 kf, NO AMB3R (*"creo que ni siquiera había amb3r en el pipeline"* — right: AMB3R entered 06-11). His order:
  *"pasá todo a ese código y dejémonos de romper las pelotas … corremos todos con ese código que andaba de
  maravilla, podemos probar ese código con el voto de f6 colmap"* → `structure.probe: omega` (no AMB3R, no coverage
  fill), `cut: covis`, `chunk_view_cap: 0` = UNCAPPED (the 13.67-length budget alone closes a chunk; Omega's
  resolution adapts to the largest chunk, ≤ 1024). Everything downstream intact (COLMAP vote, SAM3 groups, GenRecon
  anchored + 1 pt, floor). Test order: station CLOUD ONLY, then pccr 3108 CLOUD ONLY — his eye decides before the
  rest. pccr seams were 1.5–5.4 cm under the 11-08 front (healthy): the swap is judged on station without losing
  that reference.

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
- GPU PORTS (2026-11-01, *"todo debe correr en GPU, en modo determinista … la GPU no debe quedar parada nunca"*;
  again 2026-11-11, *"todo debe ser GPU nada en cpu"*, after pccr 2408's census held the card idle over an hour):
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
- ⭐ NEXT, AS SOON AS THE RECONSTRUCTION STAGE CLOSES (USER 2026-11-09, ~01:30 CDMX: "anotá para lo que sigue porque
  presentamos el proyecto en poco más de un mes … ni bien terminemos es lo que sigue"), in this order:
  1. FUSION — TWO functions (USER 2026-11-09), both needing the scans coherent in POSITION and above all in SCALE:
     a. MERGE: "la misma persona va el lunes y escanea una planta, el martes otro cuarto que no pudo escanear el lunes …
        eso no es para comparar sino que deben fusionarse en una única gran nube y mesh" — they may stay two underneath,
        but for viewing and loading they are ONE.
     b. COMPARE: pccr 2026-08-24 scanned the whole technical room, pccr 2026-08-31 only a part of it — "ahí hay que
        acoplarlas y comparar, no es en sí mismo una fusión" — aligned on the points / zones they share ("deben
        coincidir algunos puntos en común o zonas, ya veremos cómo se implementa"), then compared.
  2. BIM COMPARISON — "ya está hecha, hay que retomarla y comparar": the same as 1b with an IFC in place of the second
     scan ("es lo mismo, o bastante parecido"); maybe build a base BIM model ("por ahí armás un BIM de base, ya veremos")
     to show it.
  3. TICKETS — a simple tool to mark tickets anywhere in the project, as a review item or a non-conformity: "es para
     mostrar el concepto nomás".
  4. LATER (USER 2026-11-09: "luego, ahora no, pero luego lo pasemos a map, no solo porque parece ser mejor sino más que
     nada por la licencia, y nosotros ya tenemos map"): replace DA3 by MapAnything (its version with a commercially
     usable licence — DA3's licence is unresolved) as the pipeline's METRIC source — the focal probe (K and AMB3R's
     metre), I3's windows, Omega's scale_align, the cubes' anchors, F2's gauge. Evidence so far: KITScenes (LiDAR,
     outdoor, 0–100 m) AbsRel MapAnything 14.9 % / δ1 83 % vs DA3 27.8 % / 49 %; GeoReader (DA3NESTED-GIANT-LARGE-1.1,
     no scale alignment) 5.9–8.8 % on ScanNet / ScanNet++ / ARKitScenes; his eye: zaragoza always too BIG, pccr 3108
     too SMALL (a per-scene DA3 metre bias). Decide the source by measuring our sessions against known real dimensions.
  He expects GenRecon may need a few knobs for the right mesh ("es más rápido y es la última etapa"); after that the work
  moves to fronts other than the cloud.
- AFTER DECEMBER, IF THE PROJECT PASSES (USER 2026-11-11: "luego de diciembre, si pasa el proyecto nos centramos en
  mejorar la comprensión de la escena mucho mejor" — the VLM "está muy básico … hemos puesto todos nuestros
  esfuerzos en la geometría … ninguno era atacable si la geometría no estaba correcta"): SCENE UNDERSTANDING. SAM 3.1
  stays the recogniser (he agreed 2026-11-11: no 3D-native model replaces it — they learned from a few thousand
  rooms and every one with public weights is CC-BY-NC: SpaCeFormer, Locate 3D / 3D-JEPA, SpatialLM). The work: a 3D
  relations layer over the segmented cloud, the VLM proposing and the geometry measuring (support = contact +
  vertical order + centre of mass inside the support polygon; attached to wall / ceiling / floor; containment;
  clearances; material `vlm_proposed`), feeding the spatial chat; and a much better VLM stage. Optional witness:
  SpaCeFormer vs SAM3 on pccr 3108 (non-commercial licence).
- AFTER DECEMBER, ALSO: SPEED (USER 2026-11-11: "que pipeline robusto viejo, lento pero robusto. Insisto que luego de
  diciembre uno de los puntos a mejorar es justamente la velocidad"). Reference: pccr 2408 (2765 kf, 150 concepts)
  VLM 48.5 min + SAM3 ~10.5 h at ~4.1 min / concept even with the shared backbone (3ccb7c6); pccr 3108 (547 kf,
  87 concepts) SAM3 1 h 43 min. Robustness stays the bar: speed is never bought with a lost mask or a looser geometry.
- GENRECON INVENTS FLOOR-LEVEL SURFACE (USER 2026-11-09: "tiene eso que no sé qué es ahí abajo … en la nube ahí no hay
  nada" / "dejalo anotado, lo vemos después"). Measured on zaragoza's genrecon_photo (10 points per voxel, 5 mm cloud):
  36.5 % of the mesh area (423 of 1160 m²) has no cloud point within one GenRecon voxel (11.56 cm) — 79 m² at
  12–25 cm, 131 at 25–50, 188 at 50–100, 25 beyond 1 m; 409 m² of it between y −0.5 and +0.5 m, none above 1 m; 84 %
  of those faces have no point within 23 cm; the few points near them have a normal confidence (median 31 vs 29) —
  not noise, so neither min_points_per_voxel nor a confidence filter removes it. Main suspect: the SOLID fill of every
  support level's active columns down to the lattice base (plan_layout, USER 2026-11-08 "el suelo es sólido", against
  the double floor). Proposed test: one zaragoza GenRecon without that fill, measured the same way
  (scratchpad mesh_support.py: faces vs nearest cloud point).
- The pipeline's and the general lists: `docs/pipeline_final.md` §4 — capture distance (when he says), the weighted
  vote (his word), smarter segmentation, epoch renumbering, the GPU ports of the intake, MEJORAS_OBLIGATORIAS
  (never built), the brush lock, the BIM comparison, viewer checks; station's heavy residual tail (USER 2026-11-08
  *"si termina en mayor cobertura de la buena claro que interesa"* — measure by keyframe type / cube / region when
  the queue finishes, §4 A6).
- An evaluation against laser ground truth (ScanNet++ / ETH3D) for the IN3 and a later paper: proposed 2026-11-08,
  not ordered. USER 2026-11-11: *"más adelante lo contrastaremos contra los estándares incluso contra líderes
  profesionales"* — later, on his word. His verdict on the whole: classical photogrammetry and frontier
  feed-forward models living together, photogrammetric quality from sub-megapixel frames; the viewer *"una pc común
  en un navegador común puede navegar en una nube masiva de 114 millones de puntos con total fluidez y la carga
  inicial es instantánea"*.
- `docs/`: ARCHITECTURE.md and ROADMAP.md are out of date — update or delete on his word. His own documents
  (SCANNING_GUIDE, FUTURE_VISION, COVERAGE_STRATEGY, BIM_INTEGRATION_PLAN, the ADRs 001 / 002, FIREFOX_CONFIG, the
  phase reports signed "Ingerop IN3 Session IV") are his to keep or drop.
