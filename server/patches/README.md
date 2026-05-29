# Patches

## shaper_hf_embedder_PATCHED.py
Copia de vendor/ShapeR/model/text/hf_embedder.py CON la clase
MemoryEfficientTextFeatureExtractor reconstruida (29-05-2026).
Esa clase NO existe en el ShapeR original de Meta (era custom, se perdio
con la maquina vieja). vendor/ShapeR esta en .gitignore, por eso se guarda
aca aparte.

Al reinstalar ShapeR desde cero:
  cp server/patches/shaper_hf_embedder_PATCHED.py vendor/ShapeR/model/text/hf_embedder.py