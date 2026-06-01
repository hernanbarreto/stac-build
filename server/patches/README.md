# Patches

## shaper_hf_embedder_PATCHED.py
Copia de vendor/ShapeR/model/text/hf_embedder.py CON la clase
MemoryEfficientTextFeatureExtractor reconstruida (29-05-2026).
Esa clase NO existe en el ShapeR original de Meta (era custom, se perdio
con la maquina vieja). vendor/ShapeR esta en .gitignore, por eso se guarda
aca aparte.

Al reinstalar ShapeR desde cero:
  cp server/patches/shaper_hf_embedder_PATCHED.py vendor/ShapeR/model/text/hf_embedder.py

## VGGT-Long `--save_dir` + `--selected_frames` (01-06-2026)
El backend `mapanything` (vendor/VGGT-Long/vggt_long.py) es el fork de STAC.
Tras migrar el submódulo se perdieron dos customizaciones que `_run_mapanything`
(server/workers/map_worker.py) necesita: el stock solo acepta `--image_dir` y
`--config`, autogenera el save_dir con timestamp y corre TODOS los frames.
Reaplicar el parche en vggt_long.py + LoopModels/LoopModel.py (marcados `# STAC patch`):
  1. __init__(image_dir, save_dir, config, selected_frames=None) → self.selected_frames
  2. run(): tras el glob, si self.selected_frames, filtrar img_list por
     selected_frames.json["selected_files"] (basenames).
  3. __main__: args `--save_dir` (honra el explícito en vez del timestamp) y
     `--selected_frames`; pasarlos al constructor.
  4. run() + LoopModel.get_image_paths(): stride uniforme 1-of-N leyendo
     config['Model']['frame_stride'] — MISMO valor en ambos (loop detector + chunks)
     para que los índices queden alineados. Lo setea map_worker._build_vggt_config
     desde config.yaml reconstruction.mapanything.frame_stride.
  5. run(): escribe save_dir/frame_list.json = lista ordenada exacta de frames
     procesados (post keyframe-filter + stride). Single source of truth que usan
     map_worker (camera_frames.txt + origins frame_global REAL) y el TSDF
     (_resolve_mapanything_depth) para trazabilidad punto→frame real.
  6. process_single_chunk(): RESUME de inferencia. Al inicio, si el chunk_K.npy ya
     existe en _tmp_results_unaligned, lo carga, restaura all_camera_poses/intrinsics
     y saltea la inferencia (lo caro, ~90min). Archivo corrupto → re-infiere.
  7. apply alignment loop: NO borra unaligned incremental (rompería el resume del
     SIM3, que lee TODOS los unaligned). unaligned se borra al final (ver map_worker).
También faltaba el config base: configs/stac_mapanything.yaml (copia de
map_long_config.yaml; using_sim3=False para MapAnything; DNIO/SALAD con rutas
absolutas a los pesos del pod). VGGT-Long es submódulo → estos cambios quedan
untracked ahí; commitear en el submódulo o re-aplicar.

### Resume + disco (aligned = única copia)
- Disparar reconstrucción en modo **non-replace** → pipeline_manager NO borra
  maplong_run (línea ~247/395) e igual corre la stage → VGGT-Long resume.
- Etapas VGGT-Long (front→back): frame_list.json → loop_closures.txt → inferencia
  (_tmp_results_unaligned/chunk_K.npy, ~90min) → SIM3 (lee TODOS los unaligned) →
  apply (_tmp_results_aligned/chunk_K.npy + pcd/K_pcd.ply) → camera_poses.txt.
- aligned guarda el dict COMPLETO (solo world_points sim3-transformado; depth/conf/
  intrinsic idénticos a unaligned). Por eso el downstream lee de aligned:
  · map_worker._generate_origins → _tmp_results_aligned (conf para origins).
  · tsdf_export._resolve_mapanything_depth → _tmp_results_aligned (depth para TSDF).
- map_worker._postprocess_reconstruction borra solo _tmp_results_unaligned + _loop,
  CONSERVA _tmp_results_aligned. Corre solo si VGGT-Long terminó OK → unaligned vive
  hasta éxito total (resume a prueba de crashes; pico disco unaligned+aligned ~25G).

## MapAnything dinov2 (env mapanything) — 01-06-2026
El modelo `facebook/map-anything-apache` carga dinov2 via torch.hub HEAD, cuyo
hubconf ahora importa `dinov2.hub.cell_dino` (agregado en commit cd6f305, 17-dic-2025)
→ rompe el load. Fixes en el env `mapanything` (se pierden al reinstalar):
  1. torch: el driver del pod es CUDA 12.8; instalar build cu128 (NO cu130):
     pip install --force-reinstall torch==2.11.0 torchvision --index-url \
       https://download.pytorch.org/whl/cu128
  2. dinov2: checkout pre-cell_dino (commit 9c7e324) en /workspace/dinov2_mapanything
     (solo hubconf.py + paquete dinov2/). Patch en
     site-packages/uniception/models/encoders/dinov2.py (marcado `# STAC patch`):
     cargar con torch.hub.load("/workspace/dinov2_mapanything", model, source="local").
     Esto NO toca el cache compartido facebookresearch_dinov2_main que usa DA3.

## DA3 `--selected_frames` — SIN patch de vendor (30-05-2026)
El modo `da3` puro fallaba con "unrecognized arguments: --selected_frames"
porque el da3_streaming.py de Meta solo acepta --image_dir/--config/--output_dir.
En vez de patchear el vendor (gitignored → se pierde al reinstalar), el filtro
de keyframes vive en server/stray_da3_streaming.py::StrayDA3Streaming.run()
(override que reproduce el run() base + filtra por selected_frames.json) y el
modo da3 puro entra por server/run_da3_main.py (StrayDA3Streaming con
stray_data=None). run_da3.sh apunta a ese entry point. NO hay que tocar vendor.