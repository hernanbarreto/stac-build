# STAC - Spatio-Temporal Awareness Core

<p align="center">
  <img src="docs/assets/stac_logo.png" alt="STAC Logo" width="200"/>
</p>

<p align="center">
  <strong>Sistema de Control Dimensional y Detección de Desviaciones para Construcción</strong>
</p>

<p align="center">
  <em>Comparación en Tiempo Real: As-Built (SLAM) vs As-Planned (BIM)</em>
</p>

---

## 📋 Tabla de Contenidos

1. [Descripción del Sistema](#descripción-del-sistema)
2. [Objetivo y Alcance](#objetivo-y-alcance)
3. [Arquitectura Técnica](#arquitectura-técnica)
4. [Stack Tecnológico](#stack-tecnológico)
5. [Formatos de Archivo Soportados](#formatos-de-archivo-soportados)
6. [Detección de Desviaciones](#detección-de-desviaciones)
7. [Detección de Defectos](#detección-de-defectos)
8. [Calibración del Sistema](#calibración-del-sistema)
9. [Configuración de Alarmas y Tolerancias](#configuración-de-alarmas-y-tolerancias)
10. [Interfaz de Usuario y Control por Gestos](#interfaz-de-usuario-y-control-por-gestos)
11. [Flujo de Trabajo Operativo](#flujo-de-trabajo-operativo)
12. [Métricas y Reportes](#métricas-y-reportes)
13. [Limitaciones del Sistema](#limitaciones-del-sistema)
14. [Conformidad con Estándares](#conformidad-con-estándares)
15. [Requisitos de Hardware](#requisitos-de-hardware)
16. [Instalación y Despliegue](#instalación-y-despliegue)

---

## Descripción del Sistema

**STAC (Spatio-Temporal Awareness Core)** es un sistema de control dimensional industrial diseñado para la verificación métrica de obra construida contra modelos BIM de proyecto. El sistema permite a ingenieros de campo realizar inspecciones de control de calidad dimensional en tiempo real utilizando tecnología de Realidad Aumentada (AR) y reconstrucción 3D basada en SLAM (Simultaneous Localization and Mapping).

### Principio de Operación

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRINCIPIO STAC                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   AS-BUILT (Realidad)              AS-PLANNED (Proyecto)                    │
│   ┌──────────────────┐             ┌──────────────────┐                     │
│   │  Escaneo SLAM    │             │   Modelo BIM     │                     │
│   │  en Tiempo Real  │             │   IFC/RVT        │                     │
│   └────────┬─────────┘             └────────┬─────────┘                     │
│            │                                │                               │
│            └──────────┬─────────────────────┘                               │
│                       │                                                     │
│                       ▼                                                     │
│            ┌──────────────────────┐                                         │
│            │   COMPARACIÓN 3D    │                                         │
│            │   Punto a Punto     │                                         │
│            └──────────┬──────────┘                                         │
│                       │                                                     │
│                       ▼                                                     │
│            ┌──────────────────────┐                                         │
│            │  DESVIACIONES +     │                                         │
│            │  DEFECTOS           │                                         │
│            └──────────┬──────────┘                                         │
│                       │                                                     │
│                       ▼                                                     │
│            ┌──────────────────────┐                                         │
│            │  VISUALIZACIÓN AR   │                                         │
│            │  + REPORTES         │                                         │
│            └─────────────────────┘                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Objetivo y Alcance

### Objetivo Principal

Proporcionar una herramienta métrica de ingeniería para el **control dimensional de obra** que permita:

1. **Detectar desviaciones geométricas** entre lo construido y lo proyectado
2. **Identificar defectos constructivos** en elementos estructurales y arquitectónicos
3. **Documentar el estado de avance** de obra con precisión métrica
4. **Generar alertas en tiempo real** cuando se exceden tolerancias definidas
5. **Producir reportes auditables** para control de calidad

### Alcance de Aplicación

| Disciplina | Elementos Controlables |
|------------|------------------------|
| **Estructura** | Columnas, vigas, losas, muros estructurales, fundaciones |
| **Arquitectura** | Muros divisorios, vanos, niveles de piso terminado |
| **MEP** | Trazados principales, ubicación de equipos mayores |
| **Fachadas** | Planeidad, alineación, modulación |

### Precisión del Sistema

| Parámetro | Especificación |
|-----------|----------------|
| **Precisión posicional** | ±5 mm en condiciones óptimas |
| **Precisión angular** | ±0.1° |
| **Resolución de malla** | Configurable: 1-50 mm |
| **Rango de escaneo** | 0.3 - 10 m |
| **Velocidad de procesamiento** | 10-15 FPS (RTX 4090) |

---

## Arquitectura Técnica

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ARQUITECTURA STAC                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   CAPA INPUT    │     │   CAPA PROCESO  │     │   CAPA OUTPUT   │       │
│  ├─────────────────┤     ├─────────────────┤     ├─────────────────┤       │
│  │                 │     │                 │     │                 │       │
│  │ • AR Glasses    │────▶│ • MASt3R-SLAM   │────▶│ • AR Overlay    │       │
│  │ • Webcam RGB    │     │ • SAM3 Segment  │     │ • Web Dashboard │       │
│  │ • Video File    │     │ • IFC Parser    │     │ • PDF Reports   │       │
│  │ • LiDAR (opt)   │     │ • Comparator    │     │ • JSON/CSV Data │       │
│  │                 │     │ • Detector      │     │                 │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        CAPA DE DATOS                                │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ • Modelos BIM (.ifc, .rvt)                                          │   │
│  │ • Point Clouds (.ply, .las, .e57)                                   │   │
│  │ • Configuración de Proyecto (.yaml)                                 │   │
│  │ • Sesiones de Escaneo (SQLite)                                      │   │
│  │ • Reportes Históricos                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Módulos del Sistema

| Módulo | Responsabilidad | Librería Base |
|--------|-----------------|---------------|
| `slam_engine` | Reconstrucción 3D en tiempo real | MASt3R-SLAM |
| `segmentation` | Identificación semántica de elementos | SAM3 |
| `bim_loader` | Parsing y geometría de modelos IFC | IfcOpenShell |
| `comparator` | Alineación y cálculo de desviaciones | Open3D, ICP |
| `detector` | Clasificación de defectos | Reglas + ML |
| `ar_renderer` | Generación de overlays AR | WebXR, Three.js |
| `reporter` | Generación de reportes y métricas | ReportLab |

---

## Stack Tecnológico

### Core Libraries

```yaml
# SLAM y Reconstrucción 3D
MASt3R-SLAM: v1.0.0          # Dense SLAM con priors 3D (CVPR 2025)
lietorch: v0.3               # Lie Groups para geometría
Open3D: v0.19.0              # Procesamiento de point clouds

# Segmentación Semántica
SAM3: v1.0                   # Segment Anything Model 3 (Meta)
torch: v2.5.1+cu121          # Backend de deep learning

# BIM/IFC Processing
IfcOpenShell: v0.8.4         # Parser IFC estándar
COMPAS-IFC: v1.0             # API de alto nivel (opcional)

# Backend
Python: 3.11                 # Runtime principal
FastAPI: latest              # API REST
WebSocket: aiohttp           # Streaming en tiempo real

# Frontend AR
WebXR: latest                # Estándar AR/VR web
Three.js: latest             # Renderizado 3D
```

### Compatibilidad de Hardware

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| **GPU** | RTX 3060 (12GB) | RTX 4090 (24GB) |
| **CPU** | 8 cores | 16+ cores |
| **RAM** | 32 GB | 64 GB |
| **Storage** | SSD 500GB | NVMe 1TB+ |
| **AR Device** | HoloLens 2 | Magic Leap 2 |

---

## Formatos de Archivo Soportados

### Modelos BIM (As-Planned)

| Formato | Extensión | Soporte | Notas |
|---------|-----------|---------|-------|
| **IFC 2x3** | `.ifc` | ✅ Completo | Estándar BuildingSMART |
| **IFC 4** | `.ifc` | ✅ Completo | Recomendado |
| **IFC 4.3** | `.ifc` | ✅ Completo | Infraestructura |
| **Revit** | `.rvt` | 🔄 Vía exportación IFC | Requiere conversión |
| **ArchiCAD** | `.pln` | 🔄 Vía exportación IFC | Requiere conversión |

### Point Clouds (As-Built)

| Formato | Extensión | Soporte | Notas |
|---------|-----------|---------|-------|
| **PLY** | `.ply` | ✅ Nativo | Formato interno |
| **LAS/LAZ** | `.las`, `.laz` | ✅ Completo | LiDAR estándar |
| **E57** | `.e57` | ✅ Completo | Escáneres TLS |
| **PTS** | `.pts` | ✅ Básico | Legacy Leica |
| **XYZ** | `.xyz`, `.txt` | ✅ Básico | ASCII |

### Configuración y Reportes

| Tipo | Formato | Descripción |
|------|---------|-------------|
| Configuración | `.yaml` | Parámetros de proyecto y tolerancias |
| Sesiones | SQLite | Base de datos de escaneos |
| Reportes | PDF, HTML | Documentación de control |
| Datos | JSON, CSV | Exportación de métricas |

---

## Detección de Desviaciones

### Metodología

La detección de desviaciones se realiza mediante comparación geométrica entre la nube de puntos escaneada (As-Built) y el modelo BIM (As-Planned).

#### Algoritmo de Comparación

```
1. ALINEACIÓN INICIAL
   ├─ Registro manual de puntos de control (mínimo 3)
   ├─ Refinamiento automático con ICP (Iterative Closest Point)
   └─ Validación de RMS < umbral configurado

2. CÁLCULO DE DISTANCIAS
   ├─ Método: Point-to-Surface (más preciso)
   │   └─ Distancia de cada punto SLAM a superficie BIM más cercana
   ├─ Alternativa: Point-to-Point
   │   └─ Distancia al punto BIM más cercano (más rápido)
   └─ Resolución configurable: 1-50mm

3. CLASIFICACIÓN
   ├─ DENTRO DE TOLERANCIA (Verde)
   │   └─ |desviación| ≤ tolerancia_warning
   ├─ ADVERTENCIA (Amarillo)
   │   └─ tolerancia_warning < |desviación| ≤ tolerancia_error
   └─ FUERA DE TOLERANCIA (Rojo)
       └─ |desviación| > tolerancia_error

4. AGRUPACIÓN POR ELEMENTO
   └─ Correlación con elementos BIM via segmentación SAM3
```

### Tipos de Desviación Detectables

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **Posicional** | Elemento desplazado de su ubicación proyectada | Columna fuera de eje |
| **Dimensional** | Diferencia en dimensiones del elemento | Muro más grueso/delgado |
| **Angular** | Elemento rotado respecto a orientación proyectada | Viga inclinada |
| **Planeidad** | Superficie no plana donde debería serlo | Losa con pandeo |
| **Verticalidad** | Elemento fuera de plomo | Muro desaplomado |
| **Nivel** | Elemento en cota incorrecta | Piso a distinta altura |

### Métricas de Desviación

```yaml
# Métricas calculadas por elemento
metrics:
  mean_deviation_mm: float      # Desviación promedio
  max_deviation_mm: float       # Desviación máxima
  std_deviation_mm: float       # Desviación estándar
  percentile_95_mm: float       # Percentil 95
  points_in_tolerance: int      # Puntos dentro de tolerancia
  points_out_tolerance: int     # Puntos fuera de tolerancia
  coverage_percentage: float    # % de superficie escaneada
```

---

## Detección de Defectos

### Categorías de Defectos

#### 1. Defectos Geométricos (Detectados automáticamente)

| Defecto | Método de Detección | Severidad |
|---------|---------------------|-----------|
| **Oquedades/Cangrejeras** | Análisis de concavidad en superficie | Crítica |
| **Grietas mayores** | Discontinuidades lineales > 2mm ancho | Crítica |
| **Desplomes** | Análisis de verticalidad > tolerancia | Mayor |
| **Pandeos** | Desviación de planeidad > tolerancia | Mayor |
| **Juntas abiertas** | Gaps entre elementos > tolerancia | Menor |
| **Rebabas/Protuberancias** | Puntos con desviación positiva agrupados | Menor |

#### 2. Defectos Semánticos (Requieren segmentación SAM3)

| Defecto | Detección | Acción |
|---------|-----------|--------|
| **Elemento faltante** | En BIM pero no en SLAM | Alarma crítica |
| **Elemento adicional** | En SLAM pero no en BIM | Verificación manual |
| **Elemento mal ubicado** | Correlación espacial incorrecta | Alarma mayor |
| **Dimensiones incorrectas** | Bounding box diferente | Alarma mayor |

### Algoritmo de Detección de Defectos

```python
class DefectDetector:
    """
    Detecta defectos mediante análisis de la nube de puntos
    y comparación con geometría BIM esperada.
    """
    
    def detect_surface_defects(self, point_cloud, bim_surface):
        """
        Detecta defectos en superficie mediante:
        1. Cálculo de normales
        2. Análisis de curvatura local
        3. Detección de discontinuidades
        4. Clustering de anomalías
        """
        pass
    
    def detect_voids(self, point_cloud, expected_geometry):
        """
        Detecta oquedades/cangrejeras mediante:
        1. Voxelización del espacio
        2. Comparación de ocupación esperada vs real
        3. Identificación de regiones vacías
        """
        pass
    
    def detect_cracks(self, point_cloud):
        """
        Detecta grietas mediante:
        1. Análisis de gradientes de profundidad
        2. Detección de bordes
        3. Seguimiento de discontinuidades lineales
        """
        pass
```

---

## Calibración del Sistema

### Calibración Geométrica

#### 1. Calibración de Cámara

```yaml
camera_calibration:
  method: "checkerboard"  # o "charuco", "april_tag"
  pattern_size: [9, 6]    # Esquinas internas
  square_size_mm: 25.0    # Tamaño de cuadro
  
  # Parámetros intrínsecos resultantes
  intrinsics:
    fx: 1000.0            # Focal length X (pixels)
    fy: 1000.0            # Focal length Y (pixels)
    cx: 640.0             # Principal point X
    cy: 360.0             # Principal point Y
    
  distortion:
    k1: 0.0               # Distorsión radial
    k2: 0.0
    p1: 0.0               # Distorsión tangencial
    p2: 0.0
```

#### 2. Calibración de Escala

El sistema MASt3R-SLAM produce reconstrucciones a escala métrica. Sin embargo, se recomienda verificar la escala con:

```yaml
scale_calibration:
  method: "reference_distance"
  
  # Opción 1: Distancia conocida
  reference_points:
    point_a: [0, 0, 0]
    point_b: [1000, 0, 0]  # 1 metro de distancia conocida
    measured_mm: 1000
    
  # Opción 2: Objeto de referencia
  reference_object:
    type: "calibration_target"
    known_dimension_mm: 500
```

#### 3. Calibración de Alineación BIM

```yaml
bim_alignment:
  # Puntos de control (mínimo 3, recomendado 6+)
  control_points:
    - id: "CP01"
      bim_coords: [0, 0, 0]       # Coordenadas en modelo BIM
      field_coords: null          # Se capturan en campo
      
    - id: "CP02"
      bim_coords: [10000, 0, 0]
      field_coords: null
      
    - id: "CP03"
      bim_coords: [0, 10000, 0]
      field_coords: null
      
  # Criterios de aceptación
  acceptance:
    max_rms_mm: 10.0              # Error RMS máximo aceptable
    max_individual_error_mm: 20.0 # Error máximo por punto
```

### Procedimiento de Calibración en Campo

```
┌─────────────────────────────────────────────────────────────────┐
│                PROCEDIMIENTO DE CALIBRACIÓN                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PREPARACIÓN                                                 │
│     ├─ Identificar puntos de control en planos                  │
│     ├─ Materializar puntos de control en obra                   │
│     └─ Verificar coordenadas con estación total                 │
│                                                                 │
│  2. CALIBRACIÓN DE CÁMARA (una vez por dispositivo)             │
│     ├─ Capturar 20+ imágenes del patrón de calibración          │
│     ├─ Ejecutar calibración intrínseca                          │
│     └─ Verificar error de reproyección < 0.5 px                 │
│                                                                 │
│  3. ALINEACIÓN BIM (cada sesión o cambio de zona)               │
│     ├─ Escanear zona incluyendo puntos de control               │
│     ├─ Identificar puntos de control en nube SLAM               │
│     ├─ Calcular transformación rígida                           │
│     └─ Verificar RMS < umbral definido                          │
│                                                                 │
│  4. VERIFICACIÓN                                                │
│     ├─ Escanear elemento de referencia conocido                 │
│     ├─ Comparar dimensiones medidas vs conocidas                │
│     └─ Documentar resultados de verificación                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuración de Alarmas y Tolerancias

### Archivo de Configuración de Proyecto

```yaml
# project_config.yaml
# Configuración de tolerancias y alarmas para proyecto

project:
  name: "Edificio Torre Norte"
  code: "TN-2026"
  client: "Constructora ABC"
  
# Tolerancias por tipo de elemento (según norma aplicable)
tolerances:
  # Elementos estructurales (ACI 117, ISO 1803)
  structural:
    column:
      position_mm: 15          # Desviación de eje
      plumbness_mm_per_m: 3    # Desplome por metro
      dimension_mm: 10         # Variación dimensional
      
    beam:
      position_mm: 15
      levelness_mm_per_m: 5
      dimension_mm: 10
      
    slab:
      thickness_mm: 15
      levelness_mm_per_3m: 10  # En 3 metros
      
    wall:
      position_mm: 15
      plumbness_mm_per_m: 3
      thickness_mm: 10
      flatness_mm_per_m: 5
      
  # Elementos arquitectónicos
  architectural:
    partition_wall:
      position_mm: 20
      plumbness_mm_per_m: 5
      
    floor_finish:
      levelness_mm_per_3m: 6   # FF/FL equivalente
      
    facade:
      flatness_mm_per_m: 3
      alignment_mm: 10

# Configuración de alarmas
alarms:
  # Umbrales de clasificación
  thresholds:
    warning_factor: 0.75       # 75% de tolerancia = advertencia
    error_factor: 1.0          # 100% de tolerancia = error
    critical_factor: 1.5       # 150% de tolerancia = crítico
    
  # Acciones por nivel de alarma
  actions:
    warning:
      - log_event
      - highlight_yellow
      
    error:
      - log_event
      - highlight_red
      - notify_supervisor
      
    critical:
      - log_event
      - highlight_red_blink
      - notify_supervisor
      - notify_project_manager
      - stop_work_recommendation
      
  # Notificaciones
  notifications:
    email_recipients:
      - supervisor@proyecto.com
      - qc@proyecto.com
      
    webhook_url: "https://api.proyecto.com/stac/alerts"
    
# Reglas de negocio
business_rules:
  # Elementos que requieren verificación inmediata
  critical_elements:
    - "IfcColumn"
    - "IfcBeam"
    - "IfcFooting"
    
  # Zonas con tolerancias especiales
  special_zones:
    - zone_id: "elevator_shaft"
      tolerance_factor: 0.5    # 50% de tolerancia normal
      
    - zone_id: "facade"
      tolerance_factor: 0.75
```

### Normas de Tolerancia Incorporadas

| Norma | Aplicación | Región |
|-------|------------|--------|
| **ACI 117** | Tolerancias para hormigón | USA |
| **ACI 301** | Especificaciones estructurales | USA |
| **ISO 1803** | Tolerancias en construcción | Internacional |
| **DIN 18202** | Tolerancias dimensionales | Alemania/EU |
| **BS 8204** | Tolerancias de pisos | UK |
| **NBR 14931** | Tolerancias de hormigón | Brasil |

---

## Interfaz de Usuario y Control por Gestos

### Dispositivos de Visualización Soportados

| Dispositivo | Modo | Control |
|-------------|------|---------|
| **HoloLens 2** | AR see-through | Gestos manuales + voz |
| **Magic Leap 2** | AR see-through | Gestos manuales + controlador |
| **Meta Quest 3** | AR passthrough | Controladores + gestos |
| **Tablet/iPad** | AR passthrough | Touch + gestos |
| **Web Browser** | 3D viewer | Mouse + teclado |

### Sistema de Control por Gestos

#### Gestos Reconocidos

```
┌─────────────────────────────────────────────────────────────────┐
│                    GESTOS DE CONTROL                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✋ PALMA ABIERTA                                               │
│     └─ Pausar/Reanudar escaneo                                  │
│                                                                 │
│  👆 DEDO ÍNDICE                                                 │
│     └─ Seleccionar elemento/punto                               │
│                                                                 │
│  🤏 PINCH (Pulgar + Índice)                                     │
│     └─ Confirmar acción / Tomar medida                          │
│                                                                 │
│  👋 SWIPE HORIZONTAL                                            │
│     └─ Navegar entre elementos / Cambiar vista                  │
│                                                                 │
│  👋 SWIPE VERTICAL                                              │
│     └─ Cambiar nivel de detalle / Zoom                          │
│                                                                 │
│  ✊ PUÑO CERRADO                                                │
│     └─ Anclar/Desanclar overlay                                 │
│                                                                 │
│  🔄 ROTACIÓN DE MUÑECA                                          │
│     └─ Rotar modelo superpuesto                                 │
│                                                                 │
│  ✌️ DOS DEDOS + SEPARAR                                         │
│     └─ Zoom in                                                  │
│                                                                 │
│  ✌️ DOS DEDOS + JUNTAR                                          │
│     └─ Zoom out                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Comandos de Voz (HoloLens 2)

| Comando | Acción |
|---------|--------|
| "STAC Scan" | Iniciar escaneo |
| "STAC Stop" | Detener escaneo |
| "Show deviations" | Mostrar mapa de desviaciones |
| "Hide overlay" | Ocultar superposición BIM |
| "Take measurement" | Tomar medida manual |
| "Mark defect" | Marcar defecto manual |
| "Generate report" | Generar reporte de zona |
| "Next element" | Ir al siguiente elemento con desviación |

### Visualización AR

#### Esquema de Colores

```yaml
color_scheme:
  # Desviaciones
  within_tolerance:
    color: "#00FF00"          # Verde
    opacity: 0.3
    
  warning:
    color: "#FFFF00"          # Amarillo
    opacity: 0.5
    
  error:
    color: "#FF0000"          # Rojo
    opacity: 0.7
    
  critical:
    color: "#FF0000"          # Rojo parpadeante
    opacity: 1.0
    blink: true
    
  # Elementos BIM superpuestos
  bim_overlay:
    color: "#00BFFF"          # Azul cielo
    opacity: 0.3
    edge_color: "#FFFFFF"
    
  # Defectos
  defect_marker:
    color: "#FF00FF"          # Magenta
    shape: "sphere"
    size_mm: 50
```

#### Información en HUD (Head-Up Display)

```
┌─────────────────────────────────────────────────────────────────┐
│  STAC v2.0                              🔋 85%  📶 Connected    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Proyecto: Torre Norte                                          │
│  Zona: Nivel 5 - Sector A                                       │
│  Elemento: COL-5A-01 (IfcColumn)                                │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐     │
│  │  DESVIACIÓN DETECTADA                                 │     │
│  │                                                       │     │
│  │  Posición X: +12 mm ⚠️                                │     │
│  │  Posición Y: -3 mm ✅                                 │     │
│  │  Desplome:   8 mm/m ⚠️                               │     │
│  │                                                       │     │
│  │  [Ver Detalle]  [Marcar]  [Siguiente]                │     │
│  └───────────────────────────────────────────────────────┘     │
│                                                                 │
│  Escaneo: ████████░░ 80%    Puntos: 1.2M                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Flujo de Trabajo Operativo

### Pre-Inspección (Oficina)

```
1. PREPARACIÓN DEL MODELO
   ├─ Importar modelo IFC actualizado
   ├─ Definir zonas de inspección
   ├─ Configurar tolerancias por zona/elemento
   └─ Generar plan de puntos de control

2. CONFIGURACIÓN DEL DISPOSITIVO
   ├─ Sincronizar proyecto al dispositivo AR
   ├─ Verificar calibración de cámara
   └─ Cargar checklist de inspección
```

### Inspección en Campo

```
3. SETUP EN SITIO
   ├─ Iniciar aplicación STAC
   ├─ Seleccionar proyecto y zona
   ├─ Realizar alineación BIM (puntos de control)
   └─ Verificar calidad de alineación

4. ESCANEO
   ├─ Caminar por la zona manteniendo vista al frente
   ├─ Mantener distancia óptima (1-3m de elementos)
   ├─ Cubrir superficies desde múltiples ángulos
   └─ Verificar cobertura en tiempo real

5. REVISIÓN IN-SITU
   ├─ Revisar elementos con desviaciones detectadas
   ├─ Confirmar/descartar detecciones automáticas
   ├─ Agregar observaciones manuales
   └─ Documentar fotográficamente si necesario
```

### Post-Inspección (Oficina)

```
6. PROCESAMIENTO
   ├─ Sincronizar sesión al servidor
   ├─ Ejecutar análisis detallado (mayor resolución)
   └─ Generar nube de puntos consolidada

7. REPORTE
   ├─ Revisar resultados en dashboard web
   ├─ Generar reporte PDF automático
   ├─ Exportar datos a sistema de gestión
   └─ Archivar sesión en histórico
```

---

## Métricas y Reportes

### KPIs del Sistema

| Métrica | Descripción | Objetivo |
|---------|-------------|----------|
| **Cobertura** | % de superficie escaneada vs total | > 95% |
| **Precisión** | Error RMS vs ground truth | < 5mm |
| **Throughput** | m² escaneados por hora | > 500 m²/h |
| **Detección** | % de desviaciones reales detectadas | > 99% |
| **Falsos Positivos** | Detecciones incorrectas por sesión | < 2% |

### Contenido del Reporte de Inspección

```
┌─────────────────────────────────────────────────────────────────┐
│                  REPORTE DE INSPECCIÓN STAC                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INFORMACIÓN GENERAL                                         │
│     ├─ Proyecto, zona, fecha, inspector                         │
│     ├─ Modelo BIM de referencia (versión, fecha)                │
│     └─ Condiciones de escaneo                                   │
│                                                                 │
│  2. RESUMEN EJECUTIVO                                           │
│     ├─ Total elementos inspeccionados                           │
│     ├─ Elementos conformes / no conformes                       │
│     ├─ Defectos críticos detectados                             │
│     └─ Recomendaciones principales                              │
│                                                                 │
│  3. RESULTADOS POR ELEMENTO                                     │
│     ├─ Identificación del elemento                              │
│     ├─ Desviaciones medidas (tabla + gráfico)                   │
│     ├─ Comparación con tolerancias                              │
│     ├─ Fotografías/capturas AR                                  │
│     └─ Clasificación: CONFORME / NO CONFORME                    │
│                                                                 │
│  4. MAPA DE DESVIACIONES                                        │
│     ├─ Vista 3D con mapa de calor                               │
│     ├─ Secciones transversales                                  │
│     └─ Histograma de distribución                               │
│                                                                 │
│  5. DEFECTOS IDENTIFICADOS                                      │
│     ├─ Ubicación                                                │
│     ├─ Tipo y severidad                                         │
│     ├─ Evidencia fotográfica                                    │
│     └─ Acción recomendada                                       │
│                                                                 │
│  6. CERTIFICACIÓN                                               │
│     ├─ Firma digital del inspector                              │
│     ├─ Hash de integridad de datos                              │
│     └─ Cadena de custodia de archivos                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Limitaciones del Sistema

### Limitaciones Técnicas

| Limitación | Descripción | Mitigación |
|------------|-------------|------------|
| **Iluminación** | Requiere luz >100 lux | Luz artificial portátil |
| **Superficies reflectivas** | Vidrio, acero pulido pueden fallar | Marcadores temporales |
| **Superficies oscuras** | Negro mate absorbe luz | Iluminación adicional |
| **Movimiento** | Velocidad máxima ~1 m/s | Caminar despacio |
| **Rango** | Óptimo 0.5-5m, máximo 10m | Escaneo por secciones |
| **Oclusión** | No ve detrás de obstáculos | Múltiples pasadas |
| **Textura** | Superficies sin textura dificultan tracking | Marcadores temporales |

### Limitaciones de Precisión

| Condición | Precisión Esperada |
|-----------|-------------------|
| Condiciones óptimas | ±5 mm |
| Iluminación variable | ±10 mm |
| Superficies difíciles | ±15 mm |
| Distancias largas (>5m) | ±20 mm |
| Drift en recorridos largos | Acumulativo, requiere loop closure |

### Limitaciones Operativas

- **No reemplaza instrumentos de precisión topográfica** para control primario
- **Requiere personal capacitado** para operación e interpretación
- **Dependencia de modelo BIM actualizado** y correctamente georeferenciado
- **Requiere conectividad** para sincronización (puede operar offline temporalmente)

---

## Conformidad con Estándares

### Estándares de Construcción

| Estándar | Aplicación |
|----------|------------|
| **ISO 19650** | Gestión de información BIM |
| **ISO 12006-2** | Clasificación de información de construcción |
| **IFC 4.3** | Formato de intercambio BIM |
| **ACI 117** | Tolerancias de hormigón |
| **ASTM E1155** | Planeidad de pisos (FF/FL) |

### Estándares de Metrología

| Estándar | Aplicación |
|----------|------------|
| **ISO 17123** | Procedimientos de calibración |
| **VDI/VDE 2634** | Sistemas de medición óptica 3D |
| **ASME B89.4.19** | Performance de escáneres laser |

### Trazabilidad Metrológica

```
┌─────────────────────────────────────────────────────────────────┐
│                 CADENA DE TRAZABILIDAD                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NIST/PTB/NPL (Laboratorio Nacional)                            │
│         │                                                       │
│         ▼                                                       │
│  Estación Total Calibrada (Certificado vigente)                 │
│         │                                                       │
│         ▼                                                       │
│  Puntos de Control en Obra (Coordenadas verificadas)            │
│         │                                                       │
│         ▼                                                       │
│  Sistema STAC (Alineación a puntos de control)                  │
│         │                                                       │
│         ▼                                                       │
│  Mediciones de Desviación (Referidas a BIM alineado)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Requisitos de Hardware

### Servidor de Procesamiento

```yaml
server_requirements:
  minimum:
    gpu: "NVIDIA RTX 3060 12GB"
    cpu: "Intel i7-10700 / AMD Ryzen 7 5800X"
    ram: "32 GB DDR4"
    storage: "500 GB NVMe SSD"
    network: "Gigabit Ethernet"
    
  recommended:
    gpu: "NVIDIA RTX 4090 24GB"
    cpu: "Intel i9-13900K / AMD Ryzen 9 7950X"
    ram: "64 GB DDR5"
    storage: "2 TB NVMe SSD RAID"
    network: "10 Gigabit Ethernet"
```

### Dispositivo de Captura AR

```yaml
ar_device_requirements:
  hololens_2:
    status: "Soportado completamente"
    features:
      - "Hand tracking nativo"
      - "Eye tracking"
      - "Voice commands"
      - "Spatial anchors"
      
  magic_leap_2:
    status: "Soportado completamente"
    features:
      - "Dimming para exteriores"
      - "Mayor FOV"
      
  tablet_ipad:
    status: "Soportado (modo básico)"
    minimum: "iPad Pro M1 o superior"
    features:
      - "LiDAR opcional"
      - "ARKit"
```

---

## Instalación y Despliegue

### Instalación del Servidor

```bash
# 1. Clonar repositorio
git clone https://github.com/ingerop/stac-builder.git
cd stac-builder

# 2. Crear environment
conda create -n stac python=3.11
conda activate stac

# 3. Instalar dependencias core
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# 4. Instalar MASt3R-SLAM
cd ../mast3r_slam
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .

# 5. Descargar checkpoints
./scripts/download_checkpoints.sh

# 6. Instalar librerías BIM
pip install ifcopenshell open3d compas-ifc

# 7. Verificar instalación
python -c "import stac; stac.verify_installation()"
```

### Configuración Inicial

```bash
# 1. Inicializar base de datos
stac-cli db init

# 2. Configurar proyecto
stac-cli project create --name "Mi Proyecto" --config project_config.yaml

# 3. Importar modelo BIM
stac-cli bim import --file modelo.ifc --project "Mi Proyecto"

# 4. Iniciar servidor
stac-cli server start --port 8080
```

---

## Licencia y Soporte

### Licencia

STAC es software propietario de **Ingerop IN3**.  
Todos los derechos reservados © 2026.

### Soporte Técnico

- **Email**: soporte@stac-builder.com
- **Documentación**: https://docs.stac-builder.com
- **Actualizaciones**: https://updates.stac-builder.com

---

<p align="center">
  <strong>STAC - Spatio-Temporal Awareness Core</strong><br>
  <em>Control Dimensional Industrial para Construcción</em><br><br>
  Desarrollado por Ingerop IN3<br>
  Hernán Barreto - Session IV
</p>
