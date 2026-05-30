# Patches

## shaper_hf_embedder_PATCHED.py
Copia de vendor/ShapeR/model/text/hf_embedder.py CON la clase
MemoryEfficientTextFeatureExtractor reconstruida (29-05-2026).
Esa clase NO existe en el ShapeR original de Meta (era custom, se perdio
con la maquina vieja). vendor/ShapeR esta en .gitignore, por eso se guarda
aca aparte.

Al reinstalar ShapeR desde cero:
  cp server/patches/shaper_hf_embedder_PATCHED.py vendor/ShapeR/model/text/hf_embedder.py

## DA3 `--selected_frames` — SIN patch de vendor (30-05-2026)
El modo `da3` puro fallaba con "unrecognized arguments: --selected_frames"
porque el da3_streaming.py de Meta solo acepta --image_dir/--config/--output_dir.
En vez de patchear el vendor (gitignored → se pierde al reinstalar), el filtro
de keyframes vive en server/stray_da3_streaming.py::StrayDA3Streaming.run()
(override que reproduce el run() base + filtra por selected_frames.json) y el
modo da3 puro entra por server/run_da3_main.py (StrayDA3Streaming con
stray_data=None). run_da3.sh apunta a ese entry point. NO hay que tocar vendor.