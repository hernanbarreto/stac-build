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
- ANSWER HIM FIRST (2026-11-13, after two hours of one turn in which his "como venimos con la implementación de
  todos los puntos" went unanswered: "Contéstame … Ya mismo … Es una orden"): a question of his is answered BEFORE
  anything else; work goes in SHORT turns, one point at a time — "comités y push es en la medida que vayas
  terminando y me vas a pasar reporte cada vez que un punto quede cerrado"; the report is a few lines ("anda de a
  poco que metiste mucho texto"); a point of his list is never folded into another in a summary; no long suite or
  measurement runs in the foreground of a turn.
- THE BACKEND'S RESTART (2026-11-14: "a vos no te esta funcionando reiniciarlo, revisa que estas haciendo mal …
  sino te quedas colgado y perdemos tiempo"): only through `bash scripts/restart_backend.sh` (it waits for the
  shell, forces, starts from the prompt, waits BOUNDED) — never a Ctrl-C + sleep + an unbounded wait typed by
  hand. A restart reloads the chat's VLM (~2.5 min) and the voice (~30 s): while he is trying the app it is
  ANNOUNCED first, in one line, and server changes are batched into one restart. No probe is run against the
  live chat engine while he tests (one sequence at a time: it queues behind his questions).
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
  CPU, declared: PotreeConverter (`potree_threads: 16`, USER 2026-11-09), sqlite, PLY / JSON I/O, every JPEG encode.
  THE BACKEND NEVER TOUCHES THE CARD (found again 2026-11-13): a CUDA context in the server process is a compute
  process on the GPU for its whole life and `repro.require_exclusive_gpu` then refuses the card to every stage — GPU
  work of the server's own features runs in a child process that ENDS before a job starts (the chat's tools:
  `phase5_qa/tools_worker.py`, stopped by pipeline_manager / main when the engine lease is taken).
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
  frozen config and stops it. THE CHAT'S VLM AND THE VOICE START WITH THE BACKEND (USER 2026-11-14: "tanto el motor
  de voz y habla del sistema como el chat vlm deberan iniciar cuando levante el backend porque son el core del
  funcionamiento de stac build" — `semantic.chat_on_demand: false`, `voice.start_with_backend: true`; his
  2026-11-12 "si no lo usamos debe estar abajo" undone by it): the backend launches both at its start and after
  every GPU work, warms the VLM (its first call), and both END when a job takes the card (the voice service
  holds a CUDA context, as the chat's tools process does). init_pod.sh starts no vLLM of its own (it would be
  adopted with the stages' profile, ~3× slower for the chat). The voice: `scripts/serve_voice.sh` (env `voice`,
  built by `scripts/build_voice_env.sh`; 127.0.0.1:8798, `logs/voice_latest.log`). From a shell vLLM needs `setsid nohup`
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
  voice (Whisper large-v3-turbo MIT + Kokoro-82M Apache-2.0 + Silero VAD MIT, 2026-11-14), amb3r (AMB3R-SLAM), genrecon, CloudComPy310, gs, nodejs, colmap (evaluation only). One HF cache: `HF_HOME=
  /workspace/hf_cache`. No paid external APIs; everything local.
- texrecon (`vendor/mvs-texturing`, kept *"por las dudas"*) was relinked against libtiff 6 (its own link.txt,
  `libtbb.so.12.19` → `libtbb.so.12`); a full `cmake ..` fails on the vendored oneTBB.

## 5 · THE APP AROUND THE PIPELINE
- Reconstruir: per-scan *Segment when done* (OFF by default) and Replace (OFF = resume). Instances → Autosegment:
  the VLM prompt + SAM3 prompts (editable, saved in `output/autosegment.json`) and a checkbox per stage; GenRecon is
  its OWN button with the viewer's confidence (USER 2026-11-05). Instances → Cubes (each cube as its own box). The
  Fuse icon sits on the PROJECT row of the Sessions tab when it holds two or more scans.
- OPENING A SESSION (USER 2026-11-13: "al abrir las sesiones por defecto aparezcan las mallas genrecon apagadas para
  que abra rapido la sesion"): every mesh is listed unchecked and NOT downloaded; checking one loads it; a mesh that
  appears while the session is open loads at once.
- CALIBRATION ON THE 3D MODEL (USER 2026-11-13: "para calibrar debe hacerse tomando mediciones sobre el modelo 3d no
  sobre las imagenes porque es dificil, se debe poder seleccionar que una medicion es de calibracion y deben quedar
  almacenadas"): the distance tool, the pin of a readout → the Calibration window (real length, sigma, role gauge /
  test, source tape / laser / total station / ANOTHER SCAN) → `output/calibration_3d.json` (calibration3d.py,
  `/api/calibration/{session}`); the model's length is computed by the server from the two points and carries its
  geometry epoch (another epoch = stale, not used); the scale factor = the battery's `gauge_scale` over the gauge
  ones, with the fit's residuals and the error on the test ones. The image-anchored references (ReferenceDialog, the battery's) stay as they were.
  SINCE 2026-11-13 (97517eb), his words: "la calibracion por plano tambien debe quedar sobre la nube … y debe
  dibujar el plano sobre la nube 3d" (the plane tool: three or more picked points, Enter closes it, drawn filled);
  "ambos metodos de calibracion deben estar agrupados en un unico boton que sea calibracion, debe desplegar un
  pequeño selector" (ONE Calibration button: Distance / Plane / Stored); "debemos tener medidas de claibracion y
  de comprobacion, el tipo lo indica el usuario … sino debe indicarse la diferencia" (gauge / test; the
  verification's difference and whether it is within sigma); "debe indicarse si la nube esta calibrada o no, eso
  lo debe indicar al abrir la nube" (the HUD chip: uncalibrated / calibrated / verified); "si es por ejes, debe
  indicarse y no debe dejarse calibrar si no estan las 3 medidas, si solo hay una o dos se hace global y la otra
  medida es de verificacion" (`mode` global | axes — axes refused unless a gauge runs along each axis).
  ⭐ THE FACTOR MUST BE APPLIED, FOR REAL (USER 2026-11-13: "si" to applying it to everything measured and seen;
  then "pero ojo, el reescalado debe ser real, porque hay humano de referencia, comparacion contra otras nubes,
  debe generarse la nube real escalada … sino solo sirve localmente pero para comparar contra objetos glb de la
  coleccion, contra el humano de referencia, contra otras sesiones"): a NEW GEOMETRY of the scan at the calibrated
  scale — cloud, cameras, boxes, database, octree, meshes. BUILT 2026-11-14 (10b84e9 + the route):
  `correction/rescale.py apply_scale` — a UNIFORM scale as a new epoch through the corrections' own transaction
  (every point s·x, every camera centre s·c, the floor transform's translation × s and re-sealed, the boxes
  scaled exactly and never refit, the store, the octree) plus what an epoch does not carry (the volumes' centres,
  the placed objects' positions — a reference human keeps its real size —, the meshes by a transform at their
  root, the calibration document re-expressed so it then reads a factor of 1, the project's composition). UNDO =
  the inverse scale. `POST /api/calibration/{session}/apply` runs it in its own process (CPU only: the chat stays
  up) and the Calibration window has the button, with a confirmation. Measured on a COPY of bufferStop:
  3,837,478 points, epoch 4 → 5 in 29 s, the cloud within 0.24 µm of s·x, the octree's box × s exactly. NEVER
  RUN ON A REAL SESSION BY THE ASSISTANT: applying it is his action. PER AXIS IS NOT APPLIED TO THE GEOMETRY
  (USER 2026-11-14, to "only the global factor is applied, per axis stays a diagnosis": "si, me parece") — an
  anisotropic scale is not a similarity: the cameras cannot follow it.
- Chat = spatial intelligence (`phase5_qa`): runs in a worker thread (never on the event loop); tools for measuring
  (feature bands, vertical axis, extents), `describe_scene` (VLM, cached), persistent notes, evaluation VOLUMES
  (rest on the floor, gizmo Move / Rotate / Resize, collision tint, `measure_volume`); a session with no instance
  store gets it rebuilt from `segmentation_result.json`. SINCE 2026-11-13 (§7 THE CHAT): it answers about the scan
  ON SCREEN; its prompt holds the WHOLE inventory (by label with id ranges beyond `semantic.chat.list_limit`); the
  objects a question names (`sign#797`) are resolved on the store before the model reads it; the last
  `history_turns` turns go with the question; EVERY FIGURE of an answer is checked against the tools' results
  (provenance.py: one retry with a tool call required, what still fails is shown as unverified); the tools that
  read points run ON THE CARD in the chat's own process (`semantic.chat.tools_device: cuda`, tools_worker.py).
- ⭐ STAC SPEAKS, LISTENS AND OPERATES THE APPLICATION (USER 2026-11-14: "incorpora herramientas para hablarle y
  que traduzca a texto y que responda con voz sintetica en los 3 idiomas del sistemas, ingles, español y frances,
  el usuario debe poder hablar con stac build y stac build operar sobre todo el sistema, ampliar pantalla, usar
  todas las herramientas que tiene, preparado para todas las que tiene ahora y en un futuro … todo a traves del
  vlm … debe poder discernir entre un comando y una consulta de chat … como se integraran los del futuro que
  seran muchisimos"):
  · THE DATABASE OF COMMANDS (his "debes armar una base de datos con absolutamente todos los comandos a los que
    luego se iran incorporando los nuevos comandos"): `server/commands/` — every command is DECLARED once in
    `catalog/*.yaml` (area, kind action / query / talk, runs_on scene / system / ui, status available / manual /
    planned, arguments, and in es / en / fr its name, how people ask for it, what STAC says when done);
    `commands.db` (sqlite + FTS5) is its derived image, rebuilt when the declarations change. A NEW FUNCTION OF
    STAC = one declaration + its executor: the scene's in `phase5_qa/tools.py`, the system's in
    `commands/system_tools.py`, the interface's in `ui/src/App.tsx` — typed against
    `ui/src/commands/catalog.generated.ts` (`python -m commands.export_ui` after any change of the ui rows: a
    declared command with no executor does not compile; a test holds the file current).
  · A TURN is ONE call of the model on the candidates of the sentence (`commands/router.py`: the scene's tools
    while a scan is open + the interface's and system's found by the sentence's words in the three languages;
    conversation and help always). WHAT TELLS A COMMAND FROM A QUESTION IS THE ROW THE MODEL PICKS.
    THE DATABASE HOLDS 267 COMMANDS IN 31 AREAS (2026-11-14): 44 of the scene, 2 of the system, 46 of the
    interface that STAC runs, and the other 175 commands of the interface — every menu entry, button, toggle
    and row action inventoried in the UI (233 controls) — as `manual`: declared in the three languages, found
    by how people ask for them, and answered "se hace a mano por ahora, en <área>". Wiring one = its executor
    in App.tsx + `status: available` + its `done` text + `python -m commands.export_ui`. The
    interface's commands come back as `ui_actions`; the interface runs them and words the outcome with the
    server's texts. `manual` rows are offered too: STAC says they are done by hand. What launches GPU work or
    deletes data only OPENS ITS WINDOW by voice (a confirmed launch by voice: not built — on his word).
    The scaling path when the catalogue outgrows word search (a first call on the AREAS, 0.3 s) is declared in
    router.py, not built.
  · THE VOICE: `server/voice/` (its own process, never the backend) — speech → text with the language detected
    among es / en / fr, text → speech with one voice per language, a written answer turned into what is SAID
    (`spoken.py`). The Assistant panel: a microphone (press, speak, press), spoken answers on / off, and
    HANDS-FREE (the ear button): every utterance is written and only what names STAC is taken ("STAC, …",
    `voice/wake.py` — his "el sistema se llama stac, no hace falta llamarlo por el nombre completo stac build,
    aunque tambien se acepta"). STAC ASKS THE USER'S NAME FIRST and speaks to them by it (his "debe preguntarnos
    primero el nombre, y hablarnos usando nuestro nombre"; kept per signed-in user in the browser; a name the
    user did not say is never stored).
  · THE THIRD LANGUAGE (2026-11-14): the interface is in French too — `ui/src/i18n/fr.json` (1,296 texts, the
    same keys as es / en; a key missing there falls back to English), the glossary's French words, Settings →
    language, and `app.set_language` by voice. HIS EYE on the French: pending.
  · Measured 2026-11-14: a spoken turn after the user stops = transcription 0.3-0.4 s + the model 0.4 s (no
    scan) / 1.0-1.5 s (a scan open) + synthesis 0.1 s. HIS EAR on the three voices and HIS EYE on the commands
    in the browser: pending (nothing of the interface side was tried in a browser by the assistant).
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
- GPU, EVERYTHING THAT CAN (USER 2026-11-11: "todo lo que me marcaste que puede pasarse a GPU hay que hacerlo en GPU,
  ya lo dije mil quinientas veces. Nada debe hacerse en cpu"). Done 2026-11-11, identical results: the mask→point
  match, the audit's dilation and its mask reads in batches, the split's cloud extent, SAM3's batch IoUs and
  per-frame dedupe, the near-tie counts, and the mask store inflated ON THE CARD by nvCOMP
  (segmentation/mask_store_gpu.py; nvidia-nvcomp-cu12 6.0.0.12 installed in da3 — NVIDIA's licence, not open
  source). pccr 3108's projection: 967 s → 535 s (under the queue's contention). THE BOXES: the OBB end to end on
  the device with torch's generator for its draws, ON MAIN since 2026-11-12 (eb57ad9, his "dije ok a todo"): on
  pccr 3108 every object, point, class, fusion and audit mark identical, but 421 of 1363 boxes turned > 5° (median
  1.5°, max 45°; centre p95 8 cm, extent up to 1.6 m) — the box's yaw depends on the RANSAC draw itself for about
  a third of the objects (numpy with another seed would turn them too). His eye on the boxes is still pending; the
  way to make the yaw independent of the draw is enough RANSAC iterations for the same planes every time — to
  measure on his word.
  DONE 2026-11-12, main bcbfe3e (the night's order: "Termina de una puta vez con la implementación de los 6 puntos
  a GPU"), each measured on real sessions with the CPU function kept as the record:
  · the SAM3 mask store DEFLATED on the card (dc25e01; nvCOMP's Deflate, the CRC-32 on the device, every stream
    verified after the encode): pccr 3108 55,079 masks 81.1 → 19.6 s, pccr 2408 290,418 masks 1280.1 → 93.3 s. The
    card's store is 2.2× LARGER (65.5 → 143.8 MB, 349.2 → 771.7 MB); the server's interactive edits stay on zlib.
  · JPEG DECODE on the card (f19e34e, jpeg_gpu.py: nvJPEG's planes, libjpeg-turbo's integer upsampling and colour
    conversion on the device = PIL bit for bit): 20,279 frames of five sessions, 0 differ; 0.215 ms a frame
    against 2.485 ms; SAM3's frame loader 30–36 → 4.1–4.5 ms a frame.
  · the rounds' grey painting (de3f756): 547 + 2,765 frames, 0 images differ; 13.8 / 14.7 → 4.75 / 5.35 ms a frame.
  · GenRecon's input preparation (a7af592, `genrecon.adapter_device`): every output file byte-identical on pccr
    3108 and pccr 2408; 28.0 → 21.6 s and 121.2 → 63.5 s.
  · GenRecon's frame undistortion (bcbfe3e, `genrecon.undistort_device`): the same 916 files on bufferStop,
    observatorio and pccr 3108 — and NO TIME GAINED (≈ 3 s either way; the JPEG encode that stays weighs the most).
  NOT DONE — GenRecon's UV unwrap. Measured 2026-11-12 on three baked chunks of pccr (147–149 k faces each): 7.6 /
  16.4 / 19.1 s a chunk, 97–98.5 % of it xatlas on the CPU (add_mesh, compute_charts, pack_charts) against
  0.22–0.34 s of cumesh's clustering on the card. A GPU atlas gives OTHER UVs, so other textures: it cannot be
  held identical and needs his eye — a project, on his word.
  Still on the CPU, declared: SAM3's vendor tracker logic (Python control, the GPU ~50 % busy during SAM3), the UV
  unwrap above, every JPEG ENCODE (the VLM's views, the greyed frames, GenRecon's undistorted frames: another
  encoder writes other bytes), vLLM's own image decode; disk, sqlite, PotreeConverter. (The epoch warp and the fragment
  consolidation have device paths: correction/warp_torch.py, segmentation/fragments_torch.py.)
  THE POD'S THREADS (found 2026-11-12): `nproc` shows the host's 252 cores and the cgroup grants 26.35 CPUs — a
  torch / BLAS job left to its own thread count is throttled to a crawl (18 frames in ~170 s). Every CPU job runs
  with OMP_NUM_THREADS set (run_cloud.sh: 8; the tests: 4).
  THE SUITE (2026-11-12): with the card hidden 2,191 pass, 47 skip, 13 FAIL; the 16 files with card tests pass on
  the card (202). The 13 fail THE SAME on 89688c0 and on bcbfe3e, run alone with the card visible — not from these
  ports, not investigated (on his word): test_chunked_metric × 3, test_intake_focal × 2, test_intake_run × 3,
  test_omega_complete × 1, test_p5_orchestration × 1, test_potree_preview × 1, test_precision_runner × 2. Tests
  that need the vendor trees cannot run in a bare worktree: the suite runs in the main tree on gpu-main's commit
  (`git checkout --detach gpu-main`), then main is fast-forwarded.
- GENRECON AT SCALE (USER 2026-11-12: "arreglá genrecon y hacelo escalable que pueda manejar nubes masivas" / "la
  regla es que se continúa desde el punto de falla" / "Ok córrelo y publícalo con los tiles" / "Y aplica la
  corrección para que no vuelva a ocurrir"): pccr 2408's mesh published as tiles — 126 primitives, 16,641,666
  triangles, 251 tiles, depth 7, 1.80 GB, tiled in 618 s. 494c2e4 (the scene images DINO-encoded in batches, the
  projection's voxel batch cut to the images), eecca95 (the remesh retry at a resolution cumesh accepts; a failed
  mesh step resumes AT the mesh step), 89688c0 (the tiler never writes an empty primitive). node / WebAssembly
  never under `ulimit -v`. Known, on his word: no resume AT THE TILES step; the global remesh is computed twice;
  the stage script's `set -x` prints the HF token into the logs (never quote a log unfiltered).
- THE NIGHT'S QUEUE (USER 2026-11-12: "Cuando termines de implementar todas y queden en main listas, lanzas el
  resto de las sesiones que quedan desde el vlm al genrecon … no puede quedar el Pod sin trabajar, si alguna falla
  ocurre detienes todo corregís el error y lanzas desde el punto donde falló"): zaragoza, bufferStop, observatorio —
  VLM → SAM3 → GenRecon (confidence 0) on the cloud on disk, replace OFF, launched 2026-11-12 00:33 CDMX on main
  bcbfe3e. Volume before it: ~382 GB of the 450 GB quota (miniforge3 176, stac-build 181, hf_cache 17).
  DONE 05:14 CDMX in 4 h 41 min: the three ended ✅ with no failure and no relaunch — the first production runs of
  the five GPU ports. His eye on the GenRecon meshes, 2026-11-12: a disaster (GENRECON INVENTS FLOOR-LEVEL
  SURFACE below); the clouds and the segmentations not judged yet.
  · zaragoza (1 h 59 min): 408 segments; floor tilt 2.75°, height −74.3 cm; mesh 8,804,677 faces / 73 primitives
    in 145 tiles, depth 7, 928 MB.
  · bufferStop (59 min): 347 segments (96.87 % of 3,837,478 points); mesh 1,293,670 faces / 12 primitives in 23
    tiles, depth 4, 133 MB. The floor step turned it 92.68° on the segmented floor (904,060 pts under 174 of 174
    cameras, 0.95 m below; 92.74° on 11-08), the advisory plausibility gate failed and declared. Measured after
    it: the cameras' path is horizontal (thin axis 0.6° from y, the cameras 0.84–0.98 m over the floor) — so the
    cloud ON DISK was on its side before the floor step, against §0's "comes out STANDING" (his eye, 2026-11-11):
    not investigated, his eye says which holds in the viewer. SAM3 ran out of card memory twice on the prompt
    'rock' (after 96 of 174 frames, then 48 of 88); the pipeline's own window path closed it (781 objects).
  · observatorio (1 h 43 min): 529 segments (99.80 % of 23,374,569 points); floor tilt 3.25°, height −36.7 cm;
    mesh 9,991,750 faces / 82 primitives in 163 tiles, depth 7, 997 MB.
  Every session now holds cloud + segmentation + GenRecon mesh in tiles: pccr 2408, pccr 3108, zaragoza,
  bufferStop, observatorio. (The GenRecon meshes were deleted later that day on his word — THE SEARCH below.)
- THE PROJECTION'S AUDIT READS THE PREVIOUS PROJECTION'S FUSION MAP (found 2026-11-11, not fixed — on his word):
  `MaskAudit.obj_of` = `segmentation.erase._mask_obj_by_iid` = the FUSED view (fusion_map.json of the last
  projection), so a RE-projection audits the parts that map folded as "instance has no mask object id" (pccr 3108
  copy: 13 floor parts, 8,593 marks different). A SAM3 run deletes the map first, so SAM3 → projection is not
  affected; a projection-only re-run is (the docstring promises "a pure function of its inputs").
- GENRECON INVENTS FLOOR-LEVEL SURFACE (USER 2026-11-09: "tiene eso que no sé qué es ahí abajo … en la nube ahí no hay
  nada" / "dejalo anotado, lo vemos después"). Measured on zaragoza's genrecon_photo (10 points per voxel, 5 mm cloud):
  36.5 % of the mesh area (423 of 1160 m²) has no cloud point within one GenRecon voxel (11.56 cm) — 79 m² at
  12–25 cm, 131 at 25–50, 188 at 50–100, 25 beyond 1 m; 409 m² of it between y −0.5 and +0.5 m, none above 1 m; 84 %
  of those faces have no point within 23 cm; the few points near them have a normal confidence (median 31 vs 29) —
  not noise, so neither min_points_per_voxel nor a confidence filter removes it. Main suspect: the SOLID fill of every
  support level's active columns down to the lattice base (plan_layout, USER 2026-11-08 "el suelo es sólido", against
  the double floor). Proposed test: one zaragoza GenRecon without that fill, measured the same way
  (scratchpad mesh_support.py: faces vs nearest cloud point).
  USER 2026-11-12 (~07:57 CDMX, looking at the meshes of zaragoza, bufferStop, pccr 2408 and pccr 3108): *"fijate
  zaragoza, tiene mucha malladebajo que no hay nube ahi, pero ademas por ejemplo bufferstop donde si hay nube hay
  enormes huecos, lo mismo sucede en pccr2408 y en el 3108 lo mismo un genrecon desastroso las capas de piso, es un
  verdadero desastre, inventa partes donde no existe genera enormes huecos donde si hay nube"*. TWO defects: the
  surface invented at and under the floor (every session) and LARGE HOLES WHERE THE CLOUD IS (bufferStop, pccr 2408,
  pccr 3108) — the holes not measured yet.
  ROOT CAUSES MEASURED 2026-11-12 (USER: "avanza con las mediciones y encontra la causa raiz de todos los errores, la
  malla debe ser correcta"; scratchpad mesh_vs_cloud.py, tiles_vs_cloud.py, render_views.py — open3d CUDA search):
  (1) INVENTED = THE SOLID-GROUND FILL. 25–72 % of each published mesh lies beyond one voxel (11.56 cm) of the cloud
  (zaragoza 2233 of 3090 m², pccr 2408 1376 of 3554, pccr 3108 259 of 625, observatorio 416 of 1253, bufferStop 23 of
  90) and 92–99.8 % of that area sits in FILL voxels. plan_layout fills every column holding ANY active voxel at the
  height of any of the 3–8 support levels (not only the floor's points) down to the LATTICE base — zaragoza 1.16 m
  under the floor (a layer added because the cloud reaches −1.19 m), bufferStop / observatorio 23 cm; the fill is
  37–69 % of the active voxels. (2) HOLES = THE GENERATED SUB-VOXEL GEOMETRY. The cloud decides only which voxels are
  active; 99.5–99.8 % of the cloud has mesh within one voxel but only 54–73 % within 3 cm. From bufferStop's
  keyframes, 3–12 % of the cloud's pixels meet NO mesh (vegetation, fence: fragments) and 10–20 % meet it > 10 cm
  behind (the gravel as lumps). The decoder's own mesh.ply gives the same; fill_holes, the remesh, the per-chunk
  simplification and every level of the tiles add nothing (3D holes 0.13–0.32 % at every tile depth).
- GENRECON, THE SEARCH FOR A CORRECT MESH ON BUFFERSTOP (USER 2026-11-12, day and night CDMX), committed as a1a0b60
  and 82fbdbd. Each value with its hour and his words: config.yaml, `genrecon:`.
  · STATE 2026-11-13 (verified on disk): ALL FIVE sessions hold `tsdf/genrecon_photo` + tiles at 2.5 m · 3 points
    with the edge filter, published 2026-11-13 06:05–12:39 UTC after his "Implementamos el filtro y hacemos las
    mallas de todas las sesiones con el filtro" and "deben quedar todas las sesiones implementadas y validadas" —
    faces: bufferStop 1,963,415 · pccr 3108 13,284,219 · observatorio 19,829,073 · zaragoza 18,935,256 · pccr 2408
    33,262,077 (the joint decode by regions, 82fbdbd: `genrecon.joint_decode_max_voxels`, the resume at the decode).
    His eye on these five: not recorded yet.
  · DELETED on his word (2026-11-12, before the above): every session's GenRecon mesh of the night's queue ("elimina
    los genrecon que tenemos en todas las esiones") and the queue of the other sessions ("corta las corridas y
    elimina la cola").
  · THE SHAPE FROM THE CLOUD (1d3e806 and the uncommitted work on it: the bilateral normal filter denoise.py, the
    generative refinement T 0.5–0.7): REJECTED by his eye — "Pero no se ve nada liso" / "No tiene la calidad de
    genrecon lo que veo" / "Ninguno está bien"; "Es que la única que acepto es genrecon pero correcta no asi".
    `shape_from_cloud: false`, `solid_ground: false`; the code stays, selectable.
  · THE GENERATED SHAPE with SMALLER CUBES (only the 512 models exist: voxel = cube / 32, surface cell = cube / 512):
    "Vamos a ir avanzando por aproximaciones sucesivas, hasta dar con el mejor tamaño". His eye on 3.7 / 1.85 /
    0.925 m: "El que más se parece es el de 1.85 con algunos errores igual, entre 3.7 y 1.85"; with 2.775 and 2.31 m
    beside them: "Vamos con 2.775" → published at 2.775 m · 6 pts: "visto de frente ambas columnas que tiene están
    huecas" → 3 m · 3 pts: "Hasta ahora gana 3m 3 puntos" → 2.5 m · 3 pts: "Parece que 2.5m 3 puntos es la mejor"
    and, on 14 textured views of it alone, "Se ven espectaculares las imágenes!!!". IN FORCE, his numbers:
    `cube_size_m: 2.5` (7.81 cm voxels, 4.88 mm cells), `min_points_per_voxel: 3`; bufferStop's mesh published with
    them (1,983,731 faces, 162 MB, the stage 800.6 s; with the edge filter, a1a0b60: 1,963,415 faces, 157 MB, 35
    tiles to depth 5). The published meshes, 2.775 m · 6 | 3 m · 3 | 2.5 m · 3: mesh
    beyond 3 cm of the cloud 23.4 | 24.8 | 21.4 %, cloud beyond 3 cm of the mesh 31.1 | 32.7 | 17.3 %, holes from the
    keyframes 7.8 | 6.6 | 5.65 %.
  · FOUND, not corrected (on his word): production's mesh step changes the generated mesh — the global remesh closes
    every sheet into a shell (the floor seen as two sheets on 85–99.8 % of the rays) and the simplification to
    150,000 faces a chunk leaves the faceting. The generation is not identical run to run (seen, not investigated).
    The loose pieces stay in this family.
  · THE EDGE FILTER ON THE GENERATED MESH — NO VISIBLE GAIN, KEPT ON (USER 2026-11-12 night: "es posible aplicar el
    filtro que teníamos para las aristas?" / "Lo que quiero es que corras sobre la de 2.5m 3 puntos el afilado de
    aristas y Mostrame imágenes comparativas lado a lado con y sin filtro de aristas"; then "Lo dejamos el filtro
    igual porque podrían haber mallas que si lo requieran"): `genrecon.edge_filter: true`, denoise.py (20 / 10 /
    0.35) at Step 1.5 of the mesh step, after the global remesh, before the chunks' simplification (a1a0b60). The
    test that preceded it, on the SAME decoded mesh: 7,211,584 faces in 2 s (12.9 GB): normals turned 1.66° median (p99 64°),
    vertices moved 0.083 mm median (p99 2.3 mm, max 8 mm); the result 1,965,867 faces. Without | with: keyframe
    12's window with no mesh 7.36 | 7.37 %, keyframe 108's 1.85 | 1.84 %; the three steel plates' normal median
    1.3 | 1.3°, p90 3.3 | 4.1°, relief 2.4 | 2.3 mm. HIS EYE on four side-by-side views (textured and shaded):
    "No noto la diferencia".
  · CHAT ON DEMAND (USER 2026-11-12: "Baja al chat, no se usa" / "El chat debe cargar a demanda en esta instancia,
    si no lo usamos debe estar abajo"): `semantic.chat_on_demand: true` — vLLM is not reloaded after a stage or at
    the server's start; it loads when the chat is opened or asked. An idle auto-stop: on his value.
- THE CHAT (USER 2026-11-13: "el chat no supo encontrar elementos por ejemplo en pccr3108 no encontro sign#797,
  estaba bien escrito, ademas la cantidad de elementos que encontro eran menores a la cantidad de la lista, esta
  desactualizado, trabaja fuerte con el chat, muchas ocsas no las entiende, no las encuentra, ademas tarda
  muchisimo en ocntestar con la placa a disposición" / then "el vlm y todas sus herramientas deben funcionar en
  GPU"). ROOT CAUSES, read on his own questions of 2026-11-13 21:30–21:38 UTC (spatial_qa_logs): (1) the store was
  the FIRST scene_r.db a glob found under the project — pccr 2026-08-24's 8,777 objects while he looked at
  2026-08-31 (its sign#797 is a wall there; the counts were the other scan's); (2) the prompt listed the first 60
  objects and the model answered "ids 1 to 63"; (3) list_objects with no label = 486 kB → HTTP 400; (4) 9.9 s of
  every question building the prompt (a query per instance), and the panel re-read every object before showing an
  answer; (5) found while testing: the model stated "1.14 m, measured with get_distance" having called NO tool.
  DONE: the scan on screen (`scan` in the request, else the active scan, project_paths); the whole inventory; the
  references resolved by the tool layer; bounded listings, labels compared as people write them; the conversation's
  turns; the provenance check; the store read in single queries (prompt 9.9 → 1.4 s on pccr 2408); and the GPU
  port — phase5_qa/tools_torch.py, float64, numpy KEPT as the record, every tool identical on pccr 3108 on both
  paths: the clearance floor – wall (1.06 M × 0.74 M points) did not end in 8 min of scipy's tree, 0.09 s on the
  card (an EXACT search: octree cells with tight boxes, pruned by box distance against a real pair's distance;
  verified by an independent exact search); floor – work surface 13.3 → 0.10 s; a volume 3.3–6.7 → 0.03–0.58 s.
  The points and boxes are now promoted to float64 on BOTH paths (a display-frame store was measured in float32
  before: a figure can move in its last printed digit) — declared. His eye / use of the chat after it: pending.
  DECIDED (USER 2026-11-13: "Determinista pero en 1 o 2 segundos ninguna es negociable") — THE CHAT'S OWN ENGINE
  PROFILE: `semantic.chat.engine: single_sequence` (one sequence at a time, no batch-invariant kernels, decode
  graphs without compilation, no prefix cache, the fixed seed): the same tokens and log-probabilities bit for
  bit across request orders and across launches, 0.8 s a call; ONE call a question, the answer WRITTEN BY THE
  TOOL LAYER from the tools' results (phase5_qa/answers.py, es / en / fr) — 0.95-1.16 s a question end to end
  (2026-11-14). A stage's engine is untouched (batch_invariant; the chat-only keys do not enter its identity).
  What was measured before the decision: a question's time is now the model's. Measured 2026-11-13 on the card,
  the same two-call turn on the 9.1 k-token prompt: the deterministic serve flags (enforce-eager, no prefix cache,
  batch invariance — his 2026-09-22 "debe ser lo mas determinista posible") 21.7 tok/s and 5.6 s; without them (CUDA
  graphs + prefix caching) 57.9 tok/s and 2.0 s cold / 1.0 s with the prompt cached, the same tool call. A fast
  profile for the CHAT's engine alone (the stages launch their own, deterministic) touches the engine's identity /
  lease code: on his word. Also on his word: a "free height" tool (the model now uses get_height_profile(ceiling,
  floor): 1.66 m min on pccr 3108).
- SEGMENTS OF ONE OR A FEW POINTS (USER 2026-11-13: "en las listas de segmentaciones aparecen muchos elementos con 1
  o muy pocos puntos segmentados, teniamos un maximo de elementos"). MEASURED: every instance under
  `min_instance_points` (1000, USER 2026-09-19) is a STRUCTURAL label (door, beam, duct, pipe, wall, opening …) kept
  by `structural_always` (USER 2026-10-15 "no debe olvidarse lo estructural nunca"), each marked
  `kept_structural_under_min_points`: pccr 2408 845 of 8,777 (39 of ≤ 1 point, 142 of ≤ 10, 388 under 100), pccr
  3108 161 of 1,374 (8 / 25 / 67), bufferStop 77 of 347 (1 / 5 / 41), zaragoza 70 of 408 (2 / 14 / 30), observatorio
  60 of 529 (7 / 21 / 40). HIS ANSWER (2026-11-13): "Quién dijo que lo estructural nunca se exime quién te dijo
  quién inventó esa mierda" — his "no debe olvidarse lo estructural nunca" (2026-10-15) was never an exemption from
  the minimum: that reading was the assistant's own (commit 28aa741). THE MINIMUM HOLDS FOR EVERYTHING — done 2026-11-13 (a086804): the
  projection drops every instance under the minimum whatever its label, and `segmentation/apply_min_points.py`
  applied it to the five sessions already projected (backups in `output/_before_min_points/`, kept until he says:
  pccr 2408 7,932 elements left, pccr 3108 1,213, bufferStop 270, zaragoza 338, observatorio 469).
  THE LIST COMES FROM THE DATABASE (USER 2026-11-13: "Los objetos con menos de 1000 puntos quedan como
  unsegmented y salen de la lista y de la base de datos de hecho la lista debe ser tomada de la base de datos" —
  f218fcf: /segmentation/instances answers from scene_r.db, `source: database`; "la lista de todos debe quedar
  correctamente no solo la del 3108"). THE FILTER BELONGS AT THE ORIGIN (USER 2026-11-13: "ese filtrado de puntos
  de 1000 minimo, debe generarse en origen, ni siquiera deben volcarse a la base de datos y despues eliminarse") —
  the projection already drops them before the store is written; apply_min_points was the one-off for the past.
  A CLOUD REWRITTEN OUTSIDE THE BACKEND MUST TELL THE OPEN VIEWERS (USER 2026-11-13: "estoy seleccionando los
  objetos por ejemplo de pccr y estan desfasados … hay un desfase importantisimo … todo debe estar trazado desde
  una misma y unica fuente"): the five octrees were rebuilt under his open viewer, whose index then pointed into
  the new file — `POST /api/sessions/{id}/cloud_changed` (e3b7056); every tool that rewrites cloud products
  calls it.
  A PROJECT, ONLY ON HIS ORDER (USER 2026-11-13: "habria que ver la manera de migrar del el archivo a la base de
  datos siempre, incluso con el segmentation manager tambien, es un cambio grande, pero deberiamos dejar de
  depender del archivo y trabajar siempre con la base de datos"): the database as the single source instead of
  segmentation_result.json, the Segmentation Manager included.
- THE SCALES OF PCCR 2408 AND 3108 (USER 2026-11-13: "revisa como podriamos emparejar las escalas de pccr2408 y de
  3108 para que coincidan al compararse, sobre todo llevar la del 3108 a la del 2408 que es mas chica, deberia
  hacerse con la calibracion"). THE WAY, built: measure the same element on both scans and store it on 3108 as a
  calibration measurement whose source is the OTHER SCAN (the length read on 2408) — the factor of 3108 is then the
  one that brings it to 2408's scale, with its residuals over several elements. A FIRST LOOK, not a measurement of
  the factor: the heights of the same kinds of object (≥ 5000 points) — doors 1.814 m (236) on 2408 against 1.875 m
  (12) on 3108, ratio 1.034 [1.012, 1.077]; doors + racks 1.018 [0.999, 1.037]; racks alone 1.004 [0.82, 1.06]:
  3108 between 0 and ~7 % larger, too loose to set a factor (few objects on 3108, other segmentations). PENDING, HIS
  DECISION: where the factor is applied — the comparison's composition transform (fuse_scans meets both scans IN
  THE MIDDLE, √s each, by labels; he wants 3108 → 2408), the lengths shown, or the scan's geometry.
- BUFFERSTOP'S IMAGES ARE ROTATED (USER 2026-11-13: "las imagenes de bufferstop, estan rotadas, deben estar
  correctamente orientadas, revisa eso y acomodalas, y verifica que siempre no solo en esta sesion que es la unica
  rotada sino en las siguientes queden correctamente orientadas"). CAUSE: the video carries a rotation of 90°
  (CAP_PROP_ORIENTATION_META) and the frame extraction only RECORDED it. FIXED FOR EVERY FUTURE EXTRACTION
  (2026-11-14, `intake/frames_manifest.py`): the decoder's own rotation stays off and the container's rotation is
  applied here to every frame (`UPRIGHT`, cv2.rotate — an exact permutation, recorded in the manifest as
  `orientation_applied`). Measured on bufferStop's video: the explicit ROTATE_90_CLOCKWISE of the coded frame IS
  the decoder's own automatic rotation bit for bit; 90 / 180 / 270 held equal to it by tests on tagged videos; a
  video with no rotation is extracted as before. (The Stray capture keeps its coded frames: its depth maps are
  aligned to them.) BUFFERSTOP ITSELF (its frames on disk are the sideways ones and every product
  was computed on them): HIS DECISION 2026-11-14, between re-running it upright and turning what is shown — "solo
  gira las imagenes cuando se muestran en las poses de camara en el viewer y en el video nomas, no hace falta mas
  nada". DONE: `frames_manifest.display_rotation(scan)` (the turn that sets a scan's STORED frames upright: None
  when they were extracted upright or the video has no rotation; bufferStop → 90° clockwise, the other four
  sessions → none), `/api/sessions/{id}/frames/{file}?upright=1` (the camera pose's image in the viewer, turned
  on the way out), and the flythrough's `frame_rotation` (its 3D camera rolled by it, the vertical field of view
  taken from the frame's horizontal). The stored frames and every product are untouched; the Segmentation
  Manager and the image references still show the frames as stored (not asked). NOT TRIED IN A BROWSER. No re-run.
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
