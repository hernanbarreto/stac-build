# STAC Coverage Strategy
## Escaneo a Distancia, Oclusión Inteligente, y Cobertura Temporal

> **Autor:** Hernán Barreto — Ingerop IN3  
> **Fecha:** 2026-03-01  
> **Estado:** Diseño aprobado, pendiente implementación

---

## Índice

1. [Escaneo a Distancia (Zoom)](#1-escaneo-a-distancia-zoom)
2. [Multi-Nivel (Pisos, Subsuelos, Fosas)](#2-multi-nivel)
3. [Motor de Cobertura con Detección de Oclusión](#3-motor-de-cobertura-con-detección-de-oclusión)
4. [Multi-Source (Múltiples Operadores)](#4-multi-source)
5. [Fases de Implementación](#5-fases-de-implementación)

---

## 1. Escaneo a Distancia (Zoom)

### Problema

STAC actualmente escanea a 1-3m de distancia. Para obras grandes (tinglados, fachadas, techos altos), 
necesitamos poder escanear a distancia considerable usando el zoom óptico del celular (hasta 20x).

### Límites Prácticos

| Condición | Zoom máx. útil | Distancia máx. | Motivo del límite |
|-----------|---------------|-----------------|-------------------|
| Caminando, mano libre | 1-2x | 5-7m | Temblor + baseline insuficiente |
| Parado, codo apoyado | 3-5x | 10-15m | Temblor amplificado por zoom |
| Celular en trípode | 5-10x | 15-30m | Resolución angular de DA3 |
| Celular con gimbal | 10-15x | 25-50m | Solo precisión de DA3 |

> **El cuello de botella NO es DA3 — es el temblor de mano.**  
> A zoom 10x, un temblor de 0.5° genera 50-70px de desplazamiento entre frames → motion blur severo.

### Estrategia: Context + Detail

El operador filma un **MP4 continuo** con dos tipos de captura:

1. **Barrido de contexto** (zoom 1x, caminando): Geometría gruesa, poses, sistema de coordenadas
2. **Disparos de detalle** (zoom 3-10x, parado/apoyado): Zonas distantes con alta resolución

El sistema **detecta las transiciones de zoom automáticamente** y procesa cada segmento por separado.

### Pipeline

```
MP4 → Frame Extract → Zoom Detection → Sequence Split
       │
       ├─ Context segments (1x): pipeline normal
       │
       └─ Detail segments (Nx zoom):
           ├─ Intrinsics ajustados: f_zoom = f_base × zoom_factor
           ├─ Frame quality filter más estricto (blur_thresh × (1 + zoom/5))
           ├─ DA3 inference(images, intrinsics=K_zoom)
           └─ ICP registration detail → context → merge unificado
```

### Componentes Nuevos

- **`zoom_detector.py`**: Detecta zoom por EXIF focal_length, feature density (ORB), y/o FOV estimation
- **Sequence Splitter**: Divide frames en segmentos por zoom band
- **Intrinsics Injection**: Modifica `da3_native_wrapper.py` para pasar `intrinsics` a `model.inference()`
- **Segment Registration**: ICP + feature match entre nubes parciales

### Notas para futuro: Unity + ARKit/ARCore

Cuando se migre la captura a Unity con ARKit/ARCore:
- Los intrínsecos vendrán directamente del framework (sin estimación)
- El zoom se trackea en tiempo real desde la Camera2/ARCamera API
- Se puede enviar metadata de zoom junto con cada frame

---

## 2. Multi-Nivel

### Concepto

DA3 SLAM es **3D completo** — no asume un plano de suelo. El auto-leveling actual
(RANSAC floor detection en `alignment_manager.py`) es un post-proceso cosmético.

### Comportamiento

- **Scan continuo multi-nivel**: El operador sube escaleras, baja a fosa, etc.
  DA3 trackea las poses relativas mientras el video sea continuo. **Funciona sin cambios.**
- **Scans separados por nivel**: Cada video se procesa independientemente y se alinea
  al BIM por separado → equivalente al problema multi-source.
- **BIM como referencia**: `IfcBuildingStorey` define cotas por piso. La alineación
  scan→BIM automáticamente maneja los niveles.

---

## 3. Motor de Cobertura con Detección de Oclusión

### Problema

```
Día 1:  Pared desnuda    → scan ve 95% de superficie BIM → avance: 95%  ✅
Día 15: Pared terminada  → scan ve 100%                  → avance: 100% ✅
Día 30: Mueble adelante  → scan ve 25% (resto ocluido)   → avance: 25%  ❌ FALSO
```

Escenarios de oclusión:
- **Paredes**: muebles de cocina, mesadas, alacenas
- **Pisos**: escombros, andamios, herramientas, equipos
- **Techos**: ductos MEP, bandejas de cables instaladas después
- **Cualquier elemento**: obstrucciones temporales o permanentes

### Arquitectura: 4 Capas Integradas

```
┌──────────────────────────────────────────────────────────────┐
│          COVERAGE ENGINE (Spatio-Temporal + Occlusion)        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  CAPA 1: Occlusion Ray-Caster                                 │
│  ├─ Poses de cámara (DA3 extrinsics) → posiciones de cámara   │
│  ├─ Nube de puntos 3D (scan actual)                           │
│  ├─ Mesh BIM (IFC)                                            │
│  │                                                            │
│  │  Para cada sample en superficie BIM:                       │
│  │  ├─ Trazar rayo: mejor_cámara → sample BIM                │
│  │  ├─ Si hay puntos de nube entre cámara y BIM → OCLUIDO     │
│  │  ├─ Si rayo limpio + puntos cerca del BIM → CUBIERTO       │
│  │  └─ Si rayo limpio + sin puntos → NO CONSTRUIDO            │
│  │                                                            │
│  CAPA 2: Occlusion Classifier (SAM3 + VLM)                   │
│  ├─ SAM3: identifica QUÉ segmento ocluye                     │
│  │   → "Los puntos bloqueantes pertenecen a 'kitchen_cabinet'"│
│  ├─ VLM (InternVL3): clasifica NATURALEZA del occluder        │
│  │   ├─ Permanente: mueble, MEP, instalación → INSTALLED      │
│  │   └─ Temporal: escombro, andamio → OBSTRUCTED (re-escanear)│
│  │                                                            │
│  CAPA 3: Cumulative Coverage Store                            │
│  ├─ coverage_history.npz por elemento BIM                     │
│  │   ├─ surface_samples: (M, 3)                               │
│  │   ├─ best_coverage: bool[M] (alguna vez cubierto?)         │
│  │   ├─ best_deviation: float[M] (mejor C2M registrado)       │
│  │   ├─ occlusion_status: enum[M] per sample                  │
│  │   ├─ occluder_label: str[M] ("kitchen_cabinet" / null)     │
│  │   └─ scan_history: [{scan_id, date, coverage_pct}]         │
│  │                                                            │
│  CAPA 4: Element State Machine                                │
│  ├─ NOT_STARTED → IN_PROGRESS → COMPLETED → VERIFIED          │
│  ├─ OCCLUDED_PERMANENT → freeze last coverage value            │
│  └─ OCCLUDED_TEMPORARY → flag for re-scan                     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Métricas por Elemento BIM

| Métrica | Qué mide | Cómo se calcula |
|---------|----------|-----------------|
| `coverage_cumulative` | % superficie BIM observada alguna vez | UNIÓN histórica de todos los scans |
| `coverage_current` | % visible en el último scan | Solo puntos visibles + cubiertos |
| `occluded_pct` | % superficie bloqueada | Ray-casting detecta obstáculos |
| `occluded_permanent_pct` | % bloqueado por instalación permanente | VLM clasifica el occluder |
| `occluded_temporary_pct` | % bloqueado por obstrucción temporal | VLM clasifica el occluder |
| `effective_coverage` | Cobertura real para reporte de avance | `coverage_cumulative` (nunca baja) |
| `quality` | Calidad de lo construido vs diseño | % de puntos dentro de tolerancia C2M |
| `advance_pct` | Avance de obra para este elemento | `effective_coverage × quality` |

### Ray-Casting: Implementación

```python
def classify_bim_surface(
    camera_poses,       # DA3 extrinsics → posiciones de cámara
    scan_cloud,         # nube de puntos 3D del scan actual
    bim_mesh,           # (verts, faces) del elemento BIM
    sam3_segments,      # segmentación SAM3 (label por punto)
):
    """
    Para cada sample en la superficie BIM, clasifica:
    COVERED | OCCLUDED | NOT_BUILT | NOT_VISIBLE
    """
    # 1. Sample la superficie BIM uniformemente (ya implementado en compute_coverage_pct)
    bim_samples = sample_mesh_surface(bim_mesh, density=4)  # 4 samples/m²
    
    # 2. Para cada sample, encontrar la cámara más cercana
    #    con ángulo normal favorable (cámara mirando hacia la superficie)
    cam_positions = extract_camera_positions(camera_poses)
    
    # 3. KDTree del scan cloud para queries rápidos
    scan_tree = KDTree(scan_cloud[:, :3])
    
    for sample in bim_samples:
        best_cam = find_best_camera(sample.pos, sample.normal, cam_positions)
        
        if best_cam is None:
            sample.status = NOT_VISIBLE
            continue
        
        # 4. Ray cast: cámara → sample BIM
        ray_origin = best_cam.position
        ray_dir = normalize(sample.position - ray_origin)
        ray_length = distance(ray_origin, sample.position)
        
        # 5. Cylindrical query: puntos dentro de r=5cm del rayo
        points_along_ray = cylindrical_query(
            scan_tree, ray_origin, ray_dir,
            max_dist=ray_length * 0.9,  # 90% del camino (no cerca del BIM)
            radius=0.05                  # 5cm de radio
        )
        
        # 6. Proximity check en la superficie BIM
        near_bim_dist, _ = scan_tree.query(sample.position)
        
        if near_bim_dist < proximity_threshold:
            sample.status = COVERED
        elif len(points_along_ray) > 0:
            sample.status = OCCLUDED
            sample.occluder = get_sam3_label(points_along_ray, sam3_segments)
        else:
            sample.status = NOT_BUILT
```

### VLM Occlusion Classification

Extender `scene_analyzer.py` con prompt para clasificar oclusiones:

```yaml
# config.yaml — sección nueva
occlusion_analysis:
  enabled: true
  prompt: |
    <image>
    In this construction site scan, identify objects that are
    BLOCKING THE VIEW of walls, floors, or ceilings behind them.
    
    For each blocking object, classify:
    - "permanent": installed fixture (cabinet, countertop, MEP duct, etc.)
    - "temporary": construction debris, scaffolding, tools, materials
    
    Return JSON: [{"label": "...", "type": "permanent|temporary"}]
```

### Datos Persistentes

```
session_dir/
├─ coverage_history/
│   ├─ element_{key}_coverage.npz    ← coverage mask acumulativo por elemento
│   ├─ coverage_timeline.json        ← snapshots scan por scan
│   └─ occlusion_report.json         ← eventos de oclusión clasificados
```

---

## 4. Multi-Source

### Concepto

Múltiples operadores escanean diferentes zonas de la obra simultáneamente.
Cada scan es un video MP4 independiente.

### Modelo

```
Proyecto
├─ BIM (IFC)
├─ Scan A (operador 1, zona norte) → nube A
├─ Scan B (operador 2, zona sur)   → nube B
└─ Coverage Store acumula de todos los scans
```

### Pipeline

1. Cada operador graba MP4 localmente
2. Upload al servidor (chunked con resume, tolerante a cortes de señal)
3. Backend procesa cada scan independientemente (pipeline actual)
4. Cada scan se alinea al BIM
5. Coverage Store acumula cobertura de todos los scans
6. Vista unificada en un solo Potree

### Offline-First

- El celular graba y guarda localmente
- Cuando hay WiFi, sube al servidor (sin necesidad de conexión continua)
- `scan_meta.json` marca: operador, zona, timestamp, es multi-source o no
- El pipeline sabe si debe esperar más scans antes de generar reporte final

---

## 5. Fases de Implementación

### Prioridad 1: Motor de Cobertura (núcleo)

| # | Componente | Archivo | Descripción |
|---|-----------|---------|-------------|
| 1 | Coverage Store | `coverage_store.py` | Modelo de datos acumulativo por elemento BIM |
| 2 | Occlusion Ray-Caster | `occlusion_raycaster.py` | Ray-casting cámara→BIM con cylindrical query |
| 3 | SAM3 integration | integrar en ray-caster | Identificar qué segmento SAM3 ocluye |
| 4 | VLM classifier | `scene_analyzer.py` ext. | InternVL3 clasifica permanente vs temporal |
| 5 | Element State Machine | `coverage_store.py` | Estado derivado de coverage + occlusion |
| 6 | Pipeline integration | `bim_comparison.py` mod. | Integrar en `run_comparison()` |

### Prioridad 2: Escaneo a Distancia

| # | Componente | Archivo | Descripción |
|---|-----------|---------|-------------|
| 7 | Zoom Detector | `zoom_detector.py` | EXIF + feature density + FOV |
| 8 | Sequence Splitter | `sequence_splitter.py` | Corta en sub-sesiones por zoom band |
| 9 | Adaptive Quality | `frame_quality.py` mod. | Threshold más estricto para zoom alto |
| 10 | Intrinsics Injection | `da3_native_wrapper.py` mod. | Pasar K_zoom a DA3 inference |
| 11 | Segment Registration | `segment_registrar.py` | ICP + merge de nubes parciales |

### Prioridad 3: Multi-Source

| # | Componente | Archivo | Descripción |
|---|-----------|---------|-------------|
| 12 | Project Model | `project_manager.py` | Proyecto con N scans independientes |
| 13 | Upload Manager | backend + UI | Upload chunked con resume |
| 14 | Cloud Merge | `merge_engine.py` | Merge de nubes alineadas + dedup |
| 15 | UI Project Panel | `ProjectPanel.tsx` | Vista de proyecto multi-scan |

---

## Stack Tecnológico Utilizado

| Componente | Tecnología | Rol en esta feature |
|-----------|-----------|---------------------|
| DA3 (Depth Anything 3) | ByteDance, 2025 | Depth + poses de cámara |
| SAM3 | Meta, 2025 | Segmentación de objetos (identifica occluders) |
| InternVL3 | OpenGVLab, 2025 | Clasificación permanente vs temporal |
| CloudCompPy | Open source | Post-procesamiento de nubes |
| IfcOpenShell | Open source | Parsing de BIM (mesh, pisos, elementos) |
| KDTree (scipy) | Open source | Queries espaciales rápidas (ray-casting) |
