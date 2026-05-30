# Refinamiento de Poses de Cámara — Investigación y Plan

> Investigación profunda (estado del arte 2025-2026) sobre cómo optimizar las
> poses de cámara de DA3 / VGGT-Long de la forma más precisa y robusta posible,
> como paso **entre el loop closure y CloudComPy**. Verificación adversarial
> (3-voto) de 25 claims sobre 22 fuentes primarias; 21 confirmados, 4 refutados.
>
> Fecha: 2026-05-30. Hardware objetivo: RTX A6000 48 GB (tiempo flexible).

---

## Contexto del pipeline STAC

- Entradas disponibles: ~700 **keyframes** RGB (ver [keyframes como fuente de
  verdad]), depth **métrico denso por-frame + confianza** de DA3 streaming,
  poses **C2W** iniciales ya optimizadas por **loop closure (pose-graph SIM3,
  alineadas al primer chunk)**, intrínsecos conocidos.
- Artefactos de poses (a sobreescribir si el refinamiento mejora):
  - `output/da3_run/camera_poses.txt` → una línea por keyframe = matriz **C2W
    4×4 aplanada** (16 números).
  - PLY de poses (posiciones de cámara como puntos coloreados por chunk).
  - `intrinsic.txt`.
- Antes se usó **pycolmap** (BA de COLMAP); se busca algo mucho más robusto.
- **ARKit/LiDAR (Stray Scanner)**: ancla métrica **solo cuando el proyecto la
  tiene**. ⚠️ El proyecto de prueba `test1_hybrid_da3_lidar` es **MONOCULAR
  PURO** pese a su nombre — NO tiene datos ARKit. Para ese caso la escala viene
  del depth métrico de DA3, no de ARKit.

---

## Veredicto

Para indoor de obra (texturas pobres, repetitivas, escala métrica a preservar),
la evidencia 2025-2026 converge en:

1. **El bundle adjustment DENSO con priors de profundidad (familias b/e) es más
   robusto** que el SfM+BA clásico con features aprendidas (familia a).
2. Los **transformers feed-forward (VGGT/MASt3R, familia c)** son excelentes
   *inicializadores* pero deben rematarse con BA — y **VGGT+BA degrada o falla en
   escenas grandes** como ~700 keyframes (GitHub issue #188).
3. El **refinamiento por 3DGS (familia d)** es pulido foto-consistente final
   opcional, **no** BA primario.
4. La **escala métrica NO debe dejarse a SfM/BA clásico** (es hasta-escala):
   anclar con ARKit/LiDAR (SIM3/Umeyama) cuando exista, o usar el depth métrico
   de DA3 como regularizador de escala (enfoque ViPE) cuando no.

### Comparación de familias

| Familia | Veredicto para STAC |
|---|---|
| (a) COLMAP/GLOMAP + SuperPoint+LightGlue (hloc) | Más robusto que SIFT, pero feature-based sufre en superficies lisas/repetitivas. No es lo más robusto hoy. |
| **(b/e) Dense BA RGB-D con depth prior (DROID-SLAM / ViPE)** | **★ Recomendado.** Consume el depth métrico DA3 como término suave ponderado por confianza. Lo más robusto en low-texture. |
| (c) VGGT / MASt3R-SfM / MASt3R-SLAM | Gran init; BA naive falla a ~700 kf. MASt3R-SLAM da poses globalmente consistentes con intrínsecos conocidos. |
| (d) 3DGS bundle-adjusting (JOGS, GSplatLoc) | Pulido foto-consistente final opcional. NO primario. |

---

## Métodos clave (confirmados, con votos y fuentes)

### NVIDIA ViPE — dense BA con depth prior  *(confianza alta, 3-0)*
Dense BA sobre keyframes que fusiona **flujo denso (DROID-SLAM) + puntos
sub-pixel (cuVSLAM) + regularización por depth métrico monocular**. Corre en una
sola GPU (3-5 FPS). Resuelve la ambigüedad de escala regularizando con depth
métrico (Metric3Dv2 / UniDepthV2 / UniK3D) → trayectorias *near-metric*. Reporta
superar a VGGT y MASt3R-SLAM en TUM-RGBD indoor. **La opción más alineada con
STAC** (RGB + depth métrico denso + poses iniciales).
- Repo: `nv-tlabs/vipe` · https://github.com/nv-tlabs/vipe
- https://research.nvidia.com/labs/toronto-ai/vipe/ · arXiv:2508.10934

### Marginalized Bundle Adjustment (MBA) — mejor precisión publicada  *(alta, 3-0 / 2-1)*
Objetivo de BA inspirado en RANSAC que maximiza el AUC de la CDF de residuales
para tolerar la **alta varianza del depth monocular**. En **ETH3D: 97.3% RRA@5°,
90.2% RTA@5°**, muy por encima de MASt3R-SfM (81.2/79.7) y COLMAP (49.0/47.8).
- Repo: `ShngJZ/Marginalized-Bundle-Adjustment` · arXiv:2602.18906
- ⚠️ ETH3D es indoor **y** outdoor (no solo indoor). MBA por sí solo **no**
  garantiza escala métrica (su recuperación de escala vía afín per-frame fue
  **REFUTADA** 1-2).

### DROID-SLAM — dense BA que consume depth  *(alta, 3-0)*
Updates recurrentes de pose y depth per-pixel vía capa **Dense Bundle
Adjustment**. Aunque entrenado en video monocular, **en test admite RGB-D sin
reentrenar**: trata el depth como variable optimizable y añade un término que
penaliza la distancia al depth medido → **patrón ideal para inyectar el depth
métrico de DA3 ponderado por confianza**. ViPE construye sobre este front/backend.
- Repo: `princeton-vl/DROID-SLAM` · arXiv:2108.10869 (§3.4)

### Transformers feed-forward (VGGT / MASt3R)  *(alta, 3-0)*
VGGT (CVPR2025 Best Paper): Re10K/CO3Dv2 AUC@30 = 85.3/88.2 feed-forward vs
COLMAP+SPSG 45.2/25.3, DUSt3R 67.7/76.7, MASt3R 76.4/81.8, VGGSfMv2 78.9/83.4.
VGGT+BA llega a SOTA (IMC AUC@10 71.26→84.91) **pero falla/degrada en escenas
grandes** (issue #188). MASt3R-SfM escala lineal usando el modelo como retriever;
MASt3R-SLAM da poses globalmente consistentes con calibración conocida.
- arXiv:2409.19152 · `rmurai0610/MASt3R-SLAM`

### 3DGS pose refinement (JOGS / GSplatLoc)  *(alta, 3-0)*
**JOGS** (Oct 2025): co-optimiza Gaussianas + poses COLMAP-free alternando
rendering diferenciable y flujo óptico 3D. **GSplatLoc**: localización contra una
escena 3DGS **pre-existente** por depth-rendering (no es joint BA). Útiles como
**pulido foto-consistente final** aprovechando la GPU de 48 GB; **no** reemplazan
el BA denso.
- arXiv:2510.26117 (JOGS) · arXiv:2412.20056 / `AtticusZeller/GsplatLoc`
- ⚠️ La precisión "0.01cm en Replica" de GSplatLoc fue **REFUTADA** (0-3).

### Referencia de techo de precisión indoor
**FoundationSLAM** (AAAI 2026): ATE **0.024 m** en TUM-RGBD indoor (vs
DROID-SLAM 0.038, MASt3R-SLAM 0.030), fuerte en superficies reflectivas/low-texture.
SLAM monocular RGB-only end-to-end (no consume poses iniciales ni depth → tangencial
como *refinador*, pero marca la frontera alcanzable). arXiv:2512.25008.
**VGGT-SLAM**: optimiza sobre SL(4)/15-DoF por cámaras **no calibradas** — menos
aplicable porque STAC tiene intrínsecos conocidos. arXiv:2505.12549.

---

## Recomendación para STAC

**Motor: dense BA RGB-D estilo ViPE/DROID-SLAM.** Encaja con lo que ya hay (DA3
es un sistema derivado de DROID-SLAM). Inicializar con las poses DA3
post-loop-closure y tratar el **depth DA3 como constraint geométrico suave
ponderado por la confianza por punto** (ya guardada).

**Escala métrica:**
- Con Stray Scanner → ancla DURA: alinear la trayectoria refinada a ARKit por
  **Umeyama/SIM3** (o imponer ARKit como constraint en el BA).
- Sin iPhone (caso `test1`) → el **depth métrico de DA3** regulariza la escala
  (ViPE). *near-metric*, no perfecto.

→ Respuesta a la pregunta original: **se refinan las poses DA3; ARKit es el
ancla métrica, no el objetivo a refinar.**

**Pulido final opcional:** JOGS (3DGS BA) sobre poses ya refinadas. Solo si el BA
denso no alcanza.

### Integración propuesta — etapa `pose_refine`
Entre el fin de DA3 (loop closure) y CloudComPy. Lee `camera_poses.txt` +
`intrinsic.txt` + depth DA3 (`da3_full/`) + RGB keyframes (+ ARKit si hay) →
corre el refiner → **sobreescribe `camera_poses.txt`**, regenera el **PLY de
poses**.

**Sinergia con origins:** como cada punto tiene `frame_global`, en vez de
re-backproyectar el depth se puede **re-posicionar la nube aplicando a cada punto
el delta de su keyframe** (`T_new · T_old⁻¹`). Re-posa `chunk_NNN.ply` (y
`cleaned_cloud.ply`) de forma exacta y barata; el `lidar_complement` se beneficia
solo.

### Plan por fases
- **Fase 1** (mayor impacto / riesgo bajo): etapa `pose_refine` con dense BA
  RGB-D (backend DROID-SLAM/ViPE) seeded con poses DA3 + depth conf-weighted, +
  ancla SIM3 a ARKit cuando exista. Sobreescribe poses + PLY + re-posa la nube
  vía `frame_global`.
- **Fase 2** (opcional): pulido 3DGS (JOGS).

### Antes de invertir: medir el baseline
Correr la reconstrucción **como está hoy** y medir si el refinamiento agrega
ganancia real o marginal:
- **Con ARKit** → ATE/RPE de poses DA3 vs ARKit (tras SIM3). Pocos mm/grados →
  refinamiento marginal; drift visible → hay ganancia real.
- **Sin ARKit** (caso `test1`) → señales indirectas: drift / paredes-dobles,
  calidad de sábana / stats C2M (media, P95, pass-rate), y una
  **reprojection-consistency check** usando `frame_global`+`pixel` (reproyectar
  puntos a su keyframe y medir error).
- Guardar copia del baseline (`camera_poses.txt` + `cleaned_cloud.ply`) para el A/B.

---

## Salvedades (honestas)
1. Campo muy rápido; varios métodos son preprints bajo revisión.
2. Los números más fuertes (ViPE>VGGT/MASt3R-SLAM; MBA 97.3/90.2; FoundationSLAM
   0.024) son **auto-reportados sin reproducción independiente**.
3. ETH3D no es indoor-only → los números de MBA no son puramente indoor.
4. **VGGT+BA falla/degrada en escenas grandes** (~700 kf) — no aplicar BA naive.
5. GSplatLoc: precisión "0.01cm" **refutada** — no confiar en sus números.
6. ViPE "metric" es framing del autor → ancla ARKit/LiDAR preferible si existe.
7. Ningún paper benchmarkea exactamente el caso STAC (refinar poses DA3
   pre-optimizadas + depth DA3 + ancla ARKit) → recomendación por composición.

## Preguntas abiertas
- ¿ViPE/MBA aceptan poses iniciales como prior, o re-hacen el SLAM from-scratch?
  ¿Hay que parchear el frontend para fijarlas/regularizarlas?
- ¿Ponderación óptima del depth DA3 por confianza en el término RGB-D? ¿vs ARKit
  como constraint duro cuando ambas señales están?
- A ~700 kf en A6000: ¿BA denso global de una pasada (ViPE) vs submaps/ventana
  deslizante con ancla métrica (evita los fallos de BA en escenas grandes)?
- ¿Ganancia real del pulido 3DGS (JOGS) sobre poses ya refinadas por BA denso en
  superficies texturadas-pobres? ¿Justifica entrenar la escena 3DGS?

## Claims refutados (NO usar)
- MASt3R-SfM "robusto incluso con apenas overlap/poco movimiento" → refutado 1-2.
- MBA recupera escala vía corrección afín per-frame → refutado 1-2.
- ViPE supera a MegaSAM/VGGT/MASt3R-SLAM en pose **e** intrínsecos → refutado 1-2
  (supera en pose en TUM-RGBD indoor; la afirmación amplia no se sostiene).
- GSplatLoc 0.01cm en Replica → refutado 0-3.

## Fuentes primarias
- ViPE: https://github.com/nv-tlabs/vipe · https://research.nvidia.com/labs/toronto-ai/vipe/ · arXiv:2508.10934
- MBA: arXiv:2602.18906 · https://github.com/ShngJZ/Marginalized-Bundle-Adjustment
- DROID-SLAM: arXiv:2108.10869 · https://github.com/princeton-vl/DROID-SLAM
- VGGT: CVPR2025 (Wang et al.) · MASt3R-SfM: arXiv:2409.19152 · MASt3R-SLAM: https://github.com/rmurai0610/MASt3R-SLAM
- JOGS: arXiv:2510.26117 · GSplatLoc: arXiv:2412.20056 · https://github.com/AtticusZeller/GsplatLoc
- BAD-Gaussians: https://lingzhezhao.github.io/BAD-Gaussians/
- FoundationSLAM: arXiv:2512.25008 · VGGT-SLAM: arXiv:2505.12549
- Tooling: hloc https://github.com/cvg/Hierarchical-Localization · GLOMAP https://github.com/colmap/glomap
