# 📋 Guía de Escaneo — STAC (Spatio-Temporal Awareness Core)

**Versión**: 1.0  
**Autor**: Hernán Barreto — Ingerop IN3  
**Última actualización**: Febrero 2026

---

## 1. Introducción

Este documento establece el procedimiento estándar para la captura de video destinada a la reconstrucción 3D mediante el sistema STAC. El sistema utiliza **estimación monocular de profundidad (Reconstruction)** a partir de un video convencional de smartphone o cámara, sin necesidad de sensores LiDAR ni equipamiento especializado.

La calidad de la nube de puntos resultante depende directamente de la técnica de captura. Un escaneo correcto produce nubes densas con precisión centimétrica; un escaneo deficiente genera artefactos, huecos y errores geométricos que ningún post-procesamiento puede corregir.

---

## 2. Principio de Funcionamiento

El sistema calcula la profundidad de cada píxel mediante **parallax** — la diferencia de perspectiva entre frames consecutivos. Para que esto funcione:

> [!IMPORTANT]
> La cámara debe **trasladarse lateralmente** respecto a las superficies. La rotación pura (girar sobre el eje) **no aporta información 3D**.

```
   ✅ CORRECTO                          ❌ INCORRECTO
   (Traslación lateral)                 (Rotación pura)

   Frame 1    Frame 2                   Frame 1    Frame 2
   📷─────►  📷                          📷↻        📷
    \       / ← parallax                  |          |
     \     /                              |          |
   ───█████───                         ───█████───
      PARED                               PARED
                                       (0° parallax = 0 profundidad)
```

---

## 3. Equipamiento

### 3.1 Cámara

| Parámetro | Mínimo | Recomendado | Notas |
|-----------|--------|-------------|-------|
| Resolución | 1080p (Full HD) | 4K (3840×2160) | Mayor resolución = más puntos por metro |
| Framerate | 30 fps | 30 fps | Más fps no mejora (el sistema filtra frames redundantes) |
| Estabilización | Electrónica (EIS) | Óptica (OIS) | Reduce blur por movimiento |
| FOV | 60° | 70-90° | Gran angular captura más contexto |
| Formato | H.264 | H.265/HEVC | H.265 menor tamaño, misma calidad |

### 3.2 Iluminación

| Condición | Resultado | Acción |
|-----------|-----------|--------|
| Buena iluminación uniforme | ✅ Óptimo | — |
| Luz artificial fluorescente | ✅ Aceptable | Verificar que no genere flicker visible |
| Zonas de sombra fuerte | ⚠️ Degradado | Usar iluminación auxiliar portátil |
| Contraluz directo | ❌ Inaceptable | Evitar filmar contra ventanas/focos |
| Oscuridad total | ❌ Inaceptable | Iluminación obligatoria |

### 3.3 Accesorios recomendados

- **Estabilizador gimbal** (DJI OM, etc.) — reduce blur significativamente
- **Linterna LED portátil** — para cuartos técnicos oscuros
- **Power bank** — para sesiones largas (>30 min de video)

---

## 4. Procedimiento de Escaneo

### 4.1 Checklist Pre-Escaneo

```
□ Batería del dispositivo > 50%
□ Espacio de almacenamiento > 5 GB disponibles
□ Resolución configurada a 1080p o 4K
□ Framerate configurado a 30 fps
□ Orientación: LANDSCAPE (horizontal) — nunca vertical
□ Iluminación verificada (sin zonas completamente oscuras)
□ Estabilizador activado/montado
□ Obstrucciones removidas (puertas abiertas, objetos en el camino)
```

> [!WARNING]
> **Nunca filmar en orientación vertical (portrait).** La reconstrucción pierde el 50% del campo de visión horizontal, degradando severamente el parallax y la cobertura.

### 4.2 Patrón de Escaneo: Método de 3 Fases

#### Fase 1: Escaneo Perimetral (~60% del tiempo)

Recorrer el **borde del espacio** con la cámara apuntando hacia el interior.

```
    ┌────────────────────────────────────┐
    │              CUARTO                │
    │                                    │
    │   [Equipo]        [Panel]          │
    │                                    │
    │         [Tubería]                  │
    │                                    │
    └────────────────────────────────────┘
     ═══════════════════════════════►  Pasada 1 (Pared Sur → Este)
    ▲                                │
    │                                ▼
     ◄═══════════════════════════════  Pasada 2 (Regreso por pared opuesta)
```

**Reglas:**
- Caminar **paralelo** a las paredes, nunca directamente hacia ellas
- Mantener distancia de **1 a 2 metros** de la pared más cercana
- Velocidad: **paso normal relajado** (~0.5-1.0 m/s)
- Cámara apuntando **perpendicular** a la dirección de caminata
- En las **esquinas**: reducir velocidad, dar 2-3 pasos extra girando suavemente

#### Fase 2: Pasada Central (~20% del tiempo)

Cruzar por el **centro** del espacio, barriendo la cámara lentamente.

```
    ┌────────────────────────────────────┐
    │              CUARTO                │
    │                                    │
    │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ── ►  │  Pasada central
    │                                    │
    └────────────────────────────────────┘
```

**Objetivo:** Capturar suelo, techo, y zonas ocluidas en la pasada perimetral.

#### Fase 3: Detalle de Objetos de Interés (~20% del tiempo)

Para cada equipo, panel o elemento que requiera mayor detalle:

```
          📷 ←── 0.5-1m ──→ [EQUIPO]
         ╱                      │
        📷                      │
       ╱                        │
      📷  (Semicírculo)         │
       ╲                        │
        📷                      │
         ╲                      │
          📷                    │
```

**Reglas:**
- Acercarse a **0.5-1.0 metros** del objeto
- Caminar en **semicírculo** alrededor del objeto
- Movimiento **lento** (~0.3 m/s)
- Siempre en **traslación lateral**, nunca acercarse/alejarse en línea recta

### 4.3 Duración Estimada

| Superficie | Duración de video | Frames útiles (estimado) |
|------------|-------------------|--------------------------|
| 10-20 m² (oficina) | 1-2 minutos | 300-600 keyframes |
| 30-50 m² (sala técnica) | 3-5 minutos | 800-1500 keyframes |
| 80-120 m² (planta) | 6-10 minutos | 1500-3000 keyframes |
| >150 m² | Dividir en sesiones | Múltiples sesiones |

> [!TIP]
> Para espacios >100 m², es más eficiente dividir en **sesiones** de ~50 m² con zonas de solapamiento de ~3m entre sesiones para facilitar el registro posterior.

---

## 5. Velocidad de Movimiento

El sistema incluye un **filtro de novedad visual** (H/F Ratio) que descarta automáticamente frames sin aporte geométrico. Esto hace que la velocidad sea menos crítica, pero existen límites.

| Velocidad | Efecto | Recomendación |
|-----------|--------|---------------|
| **Estático** (<0.1 m/s) | Todos los frames descartados | ❌ Evitar pausas prolongadas |
| **Muy lento** (0.1-0.3 m/s) | ~80% frames descartados, sin daño | ⚠️ Aceptable para detalle |
| **Normal** (0.5-1.0 m/s) | ~85-95% frames descartados, óptimo | ✅ **Ideal** |
| **Rápido** (1.0-1.5 m/s) | ~70% frames descartados, riesgo de blur | ⚠️ Solo con estabilizador |
| **Muy rápido** (>1.5 m/s) | Blur excesivo, pérdida de tracking | ❌ No escanear corriendo |

---

## 6. Precisión y Error

### 6.1 Error de Profundidad

Reconstruction tiene un **error relativo de profundidad de ~10%**. Esto significa que el error crece linealmente con la distancia:

| Distancia al objeto | Error absoluto | Resolución de puntos | Aplicación típica |
|---------------------|---------------|---------------------|-------------------|
| **0.3 m** | ±3 cm | ~1 mm | Detalle de válvulas, conexiones, soldaduras |
| **0.5 m** | ±5 cm | ~2 mm | Tuberías, instrumentación |
| **1.0 m** | ±10 cm | ~3 mm | Equipos, paneles eléctricos |
| **2.0 m** | ±20 cm | ~5 mm | Paredes, estructura general |
| **3.0 m** | ±30 cm | ~8 mm | Techos, visión general |
| **5.0 m** | ±50 cm | ~15 mm | ❌ Evitar — error inaceptable |

> [!IMPORTANT]
> **Regla práctica**: Para un error máximo aceptable de ±X cm, la distancia máxima de escaneo es **10×X cm**. Ejemplo: para ±10 cm de error, no alejarse más de **1 metro**.

### 6.2 Fuentes de Error Adicionales

| Fuente | Impacto | Mitigación |
|--------|---------|------------|
| **Alineación entre chunks (SIM3)** | Acumulación de drift entre chunks | Loop closure (SALAD) lo corrige |
| **Superficies sin textura** (paredes blancas lisas) | Features débiles → pose incierta | Agregar marcadores o aceptar menor densidad |
| **Superficies reflectivas** (metal pulido, vidrio) | Profundidad errática | SOR (Statistical Outlier Removal) filtra puntos erráticos |
| **Objetos en movimiento** (personas, puertas) | Ghosting en la nube | Detener el video si hay movimiento en la escena |
| **Blur por movimiento** | Frames inutilizables | El filtro de blur los descarta automáticamente |
| **Repetición de textura** (baldosas idénticas) | Ambigüedad en matching | Mover la cámara con baseline suficiente |

### 6.3 Comparación con Otras Tecnologías

| Tecnología | Costo equipo | Precisión | Densidad | Velocidad captura |
|-----------|-------------|-----------|----------|-------------------|
| **LiDAR terrestre** (Leica, Faro) | €50,000-150,000 | ±2 mm | 10M pts/scan | 5 min/estación |
| **LiDAR portátil** (NavVis, GeoSLAM) | €20,000-50,000 | ±10-30 mm | 300K pts/s | Tiempo real |
| **Fotogrametría clásica** (MapAnything) | €0 (cámara) | ±5-20 mm | Variable | Horas de procesamiento |
| **STAC (Reconstruction mono)** | €0 (smartphone) | ±30-100 mm | 15-50M pts | Minutos de procesamiento |
| **iPhone LiDAR** (Polycam) | €1,200 (iPhone Pro) | ±10-50 mm | 1-5M pts | Tiempo real |

**Ventaja STAC**: Cero inversión en hardware, procesamiento en GPU convencional, nube ultra-densa, y segmentación con IA integrada.

---

## 7. Lo que SE DEBE hacer

| # | Práctica | Razón |
|---|----------|-------|
| 1 | Caminar **lateral** a las superficies | Maximiza el parallax geométrico |
| 2 | Mantener **distancia 1-2m** a las paredes | Balance entre cobertura y precisión |
| 3 | Filmar en **landscape** (horizontal) | Mayor campo de visión horizontal |
| 4 | Velocidad de caminata **normal** (0.5-1 m/s) | Óptimo para el filtro de novedad |
| 5 | **Solapar** pasadas (~60-70% de superposición visual) | Permite alineación robusta entre chunks |
| 6 | En **esquinas**, reducir velocidad y dar pasos extra | Evita saltos de tracking |
| 7 | Filmar objetos de interés en **semicírculo** | Captura geometría 3D completa |
| 8 | Verificar **iluminación** antes de comenzar | Features débiles en oscuridad |
| 9 | Mantener trayectoria **continua** (sin saltos) | El sistema necesita continuidad temporal |
| 10 | **Cerrar el loop** (terminar cerca del punto de inicio) | Activa loop closure para corregir drift |

---

## 8. Lo que NO SE DEBE hacer

| # | Error | Consecuencia | Severidad |
|---|-------|-------------|-----------|
| 1 | **Rotar en el lugar** sin trasladarse | Cero información de profundidad | 🔴 Crítico |
| 2 | **Caminar directo** hacia una pared | Parallax nulo en zona central | 🔴 Crítico |
| 3 | Filmar en **portrait** (vertical) | Pérdida del 50% de cobertura horizontal | 🔴 Crítico |
| 4 | **Movimiento brusco** o sacudidas | Blur + pérdida de tracking | 🔴 Crítico |
| 5 | **Tapar el lente** parcialmente con dedos | Zona muerta permanente | 🟡 Alto |
| 6 | Filmar **personas en movimiento** | Ghosting y artefactos | 🟡 Alto |
| 7 | Filmar a **más de 5 metros** esperando detalle | Error >50 cm, inútil para inspección | 🟡 Alto |
| 8 | **Pausar y retomar** el video (stop/start) | Discontinuidad temporal, falla la alineación | 🟡 Alto |
| 9 | Filmar **superficies reflectivas** de frente | Profundidad errática (se filtra parcialmente) | 🟢 Medio |
| 10 | **Cubrir el mismo lugar** muchas veces | Desperdicio de procesamiento (se filtra automáticamente) | 🟢 Bajo |

---

## 9. Casos Especiales

### 9.1 Cuartos Técnicos / Salas de Máquinas

- Priorizar pasadas **cercanas** (0.5-1m) a los equipos
- Los cuartos técnicos suelen tener **poca iluminación** — llevar linterna
- Las **tuberías** requieren semicírculos a corta distancia
- Los **paneles eléctricos** se capturan mejor con vista frontal + lateral

### 9.2 Pasillos Estrechos

- Caminar por el **centro** del pasillo, cámara mirando hacia adelante y lateralmente
- En pasillos < 2m de ancho, una sola pasada centrada puede ser suficiente
- La cámara debe **barrer** lentamente los lados (no quedarse fija mirando al frente)

### 9.3 Espacios Grandes (>50 m²)

- Dividir en **zonas de ~50 m²** con solapamiento
- Cada zona como una sesión independiente
- Mantener **3 metros de overlap** entre zonas para registro posterior
- Numerar sesiones: `sala_A_zona1`, `sala_A_zona2`, etc.

### 9.4 Escaleras / Multinivel

- Filmar de forma **continua** sin pausar
- En los escalones, la cámara sube/baja → aporta parallax vertical (bueno)
- Velocidad **reducida** en escaleras (seguridad + calidad)

---

## 10. Post-Captura

### 10.1 Verificación en Campo

Antes de abandonar el sitio, verificar:

```
□ Video reproducible sin cortes ni glitches
□ Video en landscape
□ Duración razonable para el espacio (ver tabla §4.3)
□ Sin zonas completamente oscuras en el video
□ Sin obstrucciones prolongadas del lente
```

### 10.2 Transferencia

- Transferir el video a la estación de trabajo vía **USB** (más rápido que WiFi)
- Renombrar con convención: `YYYYMMDD_ubicacion_zona.mp4`
  - Ejemplo: `20260213_cuartotecnico_zona1.mp4`
- Mantener el archivo original **sin re-codificar** (no convertir formatos)

### 10.3 Pipeline de Procesamiento

```
Video .mp4
  ↓
1. Extracción de frames (extract_frames.py)
  ↓
2. Filtro de blur (frame_quality.py)          → Elimina frames borrosos
  ↓
3. Filtro de novedad visual (frame_selector.py) → Selecciona keyframes con parallax
  ↓
4. Reconstrucción 3D (Reconstruction)                    → Genera chunks de nube de puntos
  ↓
5. Post-procesamiento (CloudComPy)            → Merge, SOR, voxel subsampling
  ↓
6. Segmentación (SAM3)                        → Detección de objetos
  ↓
  cleaned_cloud.ply + segmentation.json
```

---

## 11. Resumen Rápido (Tarjeta de Campo)

```
╔══════════════════════════════════════════════════╗
║          ESCANEO STAC — REFERENCIA RÁPIDA        ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  ✅ Landscape (horizontal)                       ║
║  ✅ Caminar lateral a las paredes                ║
║  ✅ Distancia: 1-2m (general), 0.5m (detalle)   ║
║  ✅ Velocidad: paso normal relajado              ║
║  ✅ Esquinas: lento, pasos extra                 ║
║  ✅ Objetos: semicírculo a 0.5-1m               ║
║  ✅ Cerrar el loop (volver cerca del inicio)     ║
║                                                  ║
║  ❌ No rotar sin trasladarse                     ║
║  ❌ No caminar directo hacia la pared            ║
║  ❌ No filmar en vertical (portrait)             ║
║  ❌ No correr ni hacer movimientos bruscos       ║
║  ❌ No pausar/reanudar el video                  ║
║  ❌ No filmar a >5m esperando detalle            ║
║                                                  ║
║  Precisión: ±10% de la distancia al objeto       ║
║  (a 1m → ±10cm | a 2m → ±20cm | a 0.5m → ±5cm) ║
║                                                  ║
╚══════════════════════════════════════════════════╝
```

---

*Documento generado para el sistema STAC v1.0 — Ingerop IN3*
