# El proceso de corrección

Definido por el usuario. Cada paso dice qué hace, de dónde saca el dato y en
qué archivo está el código. Un paso no se da por bueno hasta que él lo revisa.

| paso | qué hace | código | estado |
|---|---|---|---|
| **0** | **nivelar los chunks entre sí sobre el piso que comparten** | `floor_consensus.py::measure_chunks` · `visit_drift_run.py::chunk_floor_epoch` | **validado por el usuario** |
| 1 | visitas de cada objeto, desde las máscaras | `visit_drift.py::masklet_visits` | cableado |
| 2 | puntos por masklet + cadena de filtros anidados | `visit_drift.py::points_of_masklets`, `::filter_chain` | cableado |
| — | ~~comparabilidad IQR~~ | — | ELIMINADO |
| **lazo de medición, por objeto** | | | |
| 3 | medir `t` por las tres vistas del OBB | `visit_drift.py::drift_by_views` | cableado |
| 4 | visibilidad por vóxel en ese `t` → región común | `visit_drift.py::Visibility`, `::common_region` | cableado |
| 5 | re-medir sólo en la región común, hasta que `t` no se mueva | `visit_drift.py::refine_drift` | cableado |
| **filtros que necesitan `t`** | | | |
| 6 | región común = 0 → nada observado dos veces | `visit_drift.py::refine_drift` | cableado |
| 7 | desacuerdo entre las dos vistas ≤ 2 × repetibilidad | `visit_drift_run.py::measure_epoch` | cableado |
| 8 | ambigüedad de la identidad < `max_ambiguity` | `visit_drift.py::drop_ambiguous` | cableado |
| **qué se aplica** | | | |
| 9 | cada cierre es una restricción `E(d_a) − E(d_b) = t`; los del mismo tramo, el mejor determinado | `visit_drift.py::same_stretch`, `::solve_drift_curve` | cableado |
| 10 | la curva por tramos → corrección por keyframe | `visit_drift.py::solve_drift_curve` | cableado |
| 11 | nivelar el piso a y=0 (marco de visualización) | `main.py::level_floor_core` | cableado |
| 12 | **filtrado real de la nube**, por época y sobre masklets | `visit_drift.py::cloud_filter_masklets` | cableado |
| 13 | cuántas épocas se generan | `visit_drift_run.py::run` | **abierto: 4 forzosas** |

El conductor es `visit_drift_run.measure_epoch` (pasos 1 a 10, no escribe nada)
y `::one_epoch` (aplica, publica, nivela, filtra). `certify_session` llama a
`visit_drift_run.run`, así que entra por el pipeline.

**Eliminado el mismo día:** `measure()`, que tomaba las 82 instancias fusionadas
y leía las visitas del origen de los puntos; `comparability()` y sus dos
parámetros; y la copia propia del z-buffer que tenía `tools/visit_visibility.py`.

**Sin test unitario.** El usuario los rechazó explícitamente (2026-09-18:
*"los test sinteticos desde ya te digo que no, porque nunca te salen bien y te
dan resultados equivocados"*). La validación es sobre datos reales y la hace él.


## El reparto entre anclas (paso 10)

El keyframe 0 es **ancla**: el arranque es exacto. Cada corrección medida es un
nudo. Entre dos nudos se reparte la **diferencia** entre ellos, con el anterior
de ancla; antes del primer nudo, contra el ancla 0; después del último, se
extrapola.

Con nudos en 100 y 500: el tramo 100→500 reparte `corr(500) − corr(100)` anclado
en 100, y el tramo 0→100 reparte `corr(100) − 0` anclado en 0. Con un solo nudo
se propaga hacia adelante y hacia atrás contra el ancla 0.

**Cada región tiene su propia corrección y no tienen por qué parecerse** (USER
2026-09-18). Promediar varias regiones en un nudo es el error que produjo la
época 2 de pccr, peor que la 1.

## La comparabilidad IQR: eliminada (USER 2026-09-18)

*"el 3 sacalo, no existe mas"*. Comparaba las extensiones intercuartiles de las
dos copias por eje del OBB y exigía 30 % en el peor eje. Pregunta **¿son del
mismo tamaño?**, y lo que hay que saber es **¿vieron lo mismo?**:
`white_tiled_floor#18` la pasa con dos parches de tamaño idéntico que no se
tocan. La región común la reemplaza y la contesta bien.

Con ella se fueron `correction.visit_drift.min_comparability` y
`fallback_best_n`, que existía sólo para cuando ningún objeto llegaba al 30 %.
`iqr_extent` queda: es una extensión robusta y sirve por su cuenta.

El filtrado de la nube **no** es el paso 2. Va después de corregir la pose
(usuario, 2026-09-18: *"el filtrado de la nube debe ser una vez que se corrige
la pose"*) — borrar puntos contra una pose todavía mal los borra contra la
geometría equivocada.

La visibilidad **tampoco es una etapa de filtrado**: necesita `t`, y `t` sale de
la medición. Va adentro del lazo de medición, y los filtros que la usan van
después. El orden entre ellos no es libre: la ambigüedad usa `|t|` como radio,
así que si el objeto no tiene región común su `t` no mide nada y el número de
ambigüedad tampoco — por eso el 6 va antes que el 8.

---

## Paso 1 — las visitas de cada objeto

`correction/visit_drift.py::masklet_visits` + `::visits_of`

### Qué es un objeto

Los objetos son las **masklets crudas de SAM3**, de `segmentation.json` — lo
que SAM3 trackeó **antes** de fusionar. En pccr son **220**.

No son las 82 instancias de `segmentation_result.json`. La fusión junta
masklets que el matcher juzgó la misma cosa, y juntar dos copias de un objeto
en una sola instancia destruye exactamente la evidencia que esta medición
necesita: el piso de pccr, una sola instancia después de fusionar y vista
"0-215" sin cortarse nunca, son cuarenta masklets antes de fusionar, y muchas
de ésas **sí** tienen dos visitas separadas.

### Qué KF ve a cada objeto

Sale de **las máscaras**, no de los puntos: `seg_masks.npz`, clave
`f<frame>_o<oid>`, con `oid = instance_id − 1` = el campo `id` de
`segmentation.json`. Un keyframe ve al objeto cuando SAM3 lo dibujó ahí.

El espacio de frames de las claves **no se asume**: `segmentation/mask_space.py`
mide si son posiciones de keyframe o números de frame de video y traduce. En
pccr midió `keyframe POSITION over 216 keyframes`.

Una máscara guardada pero **vacía** no cuenta como ver el objeto.

### Qué es una visita

Un vector booleano de `n_kf` posiciones marcado en cada keyframe que tiene
máscara, recorrido de 0 a 215 con dos estados: entra en "viendo" en la primera
marca, sale en la primera sin marca. Cada tramo continuo que se cierra es una
visita `(kf_inicial, kf_final)`.

**Sin umbral de hueco**: un solo keyframe sin marca corta la visita.

### Salida sobre pccr, época 0

    220 masklets, 216 keyframes, 4.456 máscaras, ninguna vacía

    visitas    1    2    3    4    5   13
    masklets 150   51   13    2    2    2

    70 masklets con 2 o más visitas

Los de dos visitas inicio↔final, que es el revisit que importa:

    white_tiled_floor#13     0-13   195-215      white_tiled_floor#5     0-16  205-215
    white_tiled_floor#20     0-16   194-212      white_tiled_floor#6     0-16  206-215
    white_tiled_floor#16     0-17   202-215      white_tiled_floor#7     0-16  206-215
    white_tiled_floor#12     0-7    195-215      white_tiled_floor#15    0-11  202-215
    desk#203                 0-12   200-215      white_tiled_floor#25    0-17  203-210
    computer_monitor#144     0-11   200-215      glass_door#121          0-13  204-215
    glass_door#119           0-5    193-213      glass_door#117          0-16  210-215
    computer_monitor#143     0-3    190-212      glass_door#118          0-17  211-215
    desk#202                 0-5    193-213      black_office_chair#199  0-6   200-215
    glass_door#120           0-4    193-212      desk#201                0-18  213-215

### Código

```python
@dataclass
class Masklet:
    oid: int
    instance_id: int
    label: str
    keyframes: np.ndarray             # posiciones de keyframe con máscara
    visits: List[Tuple[int, int]]     # (kf_primero, kf_último) por tramo


def masklet_visits(output_dir, log=print) -> List[Masklet]:
    doc = json.loads((output_dir / "segmentation.json").read_text())
    masks = np.load(output_dir / doc.get("mask_file", "seg_masks.npz"))
    space = mask_space.resolve(output_dir, masks=masks, log=log)
    n_kf = len(mask_space.keyframe_numbers(output_dir))

    kf_of_mask_frame = {int(space.from_keyframe(k)): k for k in range(n_kf)
                        if space.from_keyframe(k) is not None}

    seen = {}
    for key in masks.files:                       # f<N>_o<OID>
        m = re.match(r"^f(\d+)_o(\d+)$", key)
        if not m:
            continue
        kf = kf_of_mask_frame.get(int(m.group(1)))
        if kf is None or not masks[key].any():    # vacía = no la vio
            continue
        seen.setdefault(int(m.group(2)), set()).add(kf)

    return [Masklet(oid=int(e["id"]), instance_id=int(e["instance_id"]),
                    label=str(e["label"]), keyframes=kfs,
                    visits=visits_of(kfs, n_kf))
            for e in doc["instances"]
            for kfs in [np.array(sorted(seen.get(int(e["id"]), ())), np.int64)]]


def visits_of(ks: np.ndarray, n_kf: int) -> List[Tuple[int, int]]:
    present = np.zeros(int(n_kf), bool)
    present[np.unique(ks[ks >= 0])] = True
    out, start = [], None
    for k in range(int(n_kf)):
        if present[k] and start is None:
            start = k
        if not present[k] and start is not None:
            out.append((start, k - 1))
            start = None
    if start is not None:
        out.append((start, int(n_kf) - 1))
    return out
```

### Errores corregidos en este paso

- Tomaba las **82 instancias fusionadas** de `segmentation_result.json` en vez
  de los **220 masklets** de `segmentation.json` (usuario, 2026-09-18).
- Sacaba los keyframes del **origen de los puntos** (`frame_global`) en vez de
  **las máscaras**.


---

## Paso 2 — la cadena de filtros anidados

`correction/visit_drift.py::filter_chain`, y antes `::points_of_masklets`

### Los puntos de cada masklet

Los `globalIndices` de la nube están por instancia **fusionada**, así que un
masklet no tiene puntos propios hasta que se los pide. Cada punto lleva el
keyframe donde nació y el píxel donde nació, y la máscara del masklet en ese
keyframe dice si ese píxel está adentro. Es una consulta, no una inferencia.

Las dos grillas son distintas y no se asumen:

- píxeles de la nube: grilla de **trace**, leída de `intrinsic.txt`
  (`cx`, `cy` → `W = 2·cx`, `H = 2·cy`). En pccr **688×384**.
- máscaras: **832×464**.
- escala 1,2093. Si las dos relaciones de aspecto no coinciden dentro del 1 %,
  falla en voz alta en vez de leer el píxel equivocado.

Los masklets se solapan y nadie fuerza un ganador: un punto puede pertenecer a
varios. En pccr, **203 de 220** masklets llevan puntos, 24.875.434 asignaciones
sobre 22.128.324 puntos. 8 segundos.

### La cadena, y por qué es anidada

Cada regla se aplica a lo que dejó la anterior, y **toda regla que elimina una
visita obliga a volver a contar los objetos**: un objeto que queda con una sola
visita no puede mostrar duplicado.

1. más de 500 puntos
2. dos o más visitas
3. eliminar cada visita que está a **1 m o menos de recorrido caminado** de la
   anterior — la visita, no el objeto — y volver a contar
4. eliminar cada visita que aporta **1 % o menos** de los puntos del objeto —
   otra vez la visita — y volver a contar

El aporte de una visita es la fracción de los puntos del masklet cuyo keyframe
de nacimiento cae en esa visita, sobre el total del masklet.

### Salida sobre pccr, época 0 (recorrido total 19,32 m)

    0) masklets                              220
    1) con más de 500 puntos                 183
    2) de ésos, con 2 o más visitas           64
    3) visitas eliminadas (<= 1 m)            40   ->  54 objetos  (caen 10)
    4) visitas que aportan <= 1 %             36   ->  32 objetos  (caen 22)

El filtro de aporte es el que más corta. Los que declaran:

    objeto                     puntos   visitas kf (aporte)          recorrido
    white_tiled_floor#17       31.562   0-17(82%)   213-215(18%)       18,0 m
    desk#201                   42.078   0-18(94%)   213-215(6%)        18,0
    glass_door#117             20.026   0-16(66%)   210-215(34%)       17,9
    white_tiled_floor#14       21.666   0-13(58%)   205-215(42%)       17,6
    black_office_chair#199      8.290   0-6(6%)     200-215(94%)       17,6
    white_tiled_floor#30       67.606   0-8(38%)    201-213(62%)       17,5
    glass_door#121             58.105   0-13(62%)   204-215(38%)       17,5
    white_tiled_floor#15       74.390   0-11(64%)   202-215(36%)       17,4
    computer_monitor#144       11.720   0-11(10%)   200-215(90%)       17,3
    white_tiled_floor#25      106.365   0-17(80%)   203-210(20%)       17,2
    desk#203                   99.612   0-12(35%)   200-215(65%)       17,1
    desk#202                   55.531   0-5(4%)     193-213(96%)       16,8
    white_tiled_floor#13       39.183   0-13(26%)   195-215(74%)       16,5

**El piso entra como testigo.** Fusionado era una sola instancia vista "0-215"
sin cortarse y no declaraba nada; como masklets son cuarenta, y la mayoría de
los que sobreviven la cadena son suyos, con repartos más parejos (80/20, 58/42,
49/51) que los muebles.

### Código

```python
a = [m for m in masklets if len(points_by_oid.get(m.oid, ())) > min_points]
b = [m for m in a if m.n_visits >= 2]

kept = {}
for m in b:                                   # 3) visitas demasiado juntas
    v = [m.visits[0]]
    for cur in m.visits[1:]:
        if chainage[cur[0]] - chainage[v[-1][1]] <= min_walk_m:
            steps["visits_dropped_close"] += 1
        else:
            v.append(cur)
    kept[m.oid] = v
c = [m for m in b if len(kept[m.oid]) >= 2]   # volver a contar

out = []
for m in c:                                   # 4) visitas que no aportan
    ks, total = ks_of_point[points_by_oid[m.oid]], len(points_by_oid[m.oid])
    v, sh = [], []
    for (x, y) in kept[m.oid]:
        frac = ((ks >= x) & (ks <= y)).sum() / total
        if frac <= min_visit_share:
            steps["visits_dropped_share"] += 1
        else:
            v.append((x, y)); sh.append(frac)
    if len(v) < 2:                            # volver a contar
        continue
    out.append(Candidate(m, points_by_oid[m.oid], v, sh,
                         [chainage[v[i+1][0]] - chainage[v[i][1]]
                          for i in range(len(v) - 1)]))
```


---

## Paso 2b — la ambigüedad de la identidad

`correction/visit_drift.py::ambiguity` + `::drop_ambiguous`

### El problema que resuelve

Dos visitas de un mismo masklet pueden coincidir **perfectamente en forma** y
ser dos pedazos **distintos** de una superficie repetida.

Medido sobre pccr: de 23 categorías, `white_tiled_floor` tiene **85 masklets**
— el 39 % de la sesión — y **20 de los 32** que pasan la cadena son suyos. La
mejor correlación de silueta de toda la sesión (0,93, `white_tiled_floor#15`)
es una baldosa. Elegir por acuerdo de forma elige una baldosa.

Las tres vistas de los más "parejos" lo muestran sin discusión:

    white_tiled_floor#18   las dos copias no se tocan: 1,5 m de separación en L
    white_tiled_floor#16   la visita 1 son DOS parches, la visita 2 uno solo
    white_tiled_floor#20   3,4 m de separación entre copias de 0,8 m de lado

### La medida

    ambigüedad = cuántos OTROS masklets de la MISMA etiqueta tienen puntos
                 a menos de |t| de la primera copia de este objeto

Distancia de punto más cercano a punto más cercano, no de centroide a
centroide: un objeto extendido (una pared, una tira de piso) no tiene centro
con sentido, y lo que lo vuelve candidato a confusión es que **parte** de él
esté donde el corrimiento dice que este objeto pudo haber venido.

**No sabe qué es un piso.** Es la etiqueta contra sí misma: cuántos hay y
dónde.

### Dónde va en el orden

Necesita `|t|`, que sale de comparar las siluetas en las tres vistas. Así que
es un filtro de la cadena pero corre **después** de la medición, no antes.

### El umbral

`correction.visit_drift.max_ambiguity: 2` — pasa el objeto con **menos de 2**
rivales. Medido: los 20 pisos puntúan entre **17 y 59**; los no-piso entre
**0 y 6**. No hay nada en el medio.

### Salida sobre pccr, época 0

    objeto                     |t| cm   ambig  rivales
    glass_door#121               25,2       0   -
    exposed_ceiling_grid#169    108,8       0   -
    black_office_chair#199       66,0       0   -
    desk#201                     82,6       0   -
    desk#202                     77,1       0   -
    desk#203                     70,8       0   -
    glass_door#117               63,7       1   glass_door#118
    computer_monitor#144         88,6       1   computer_monitor#142

    8 de 32.  Los 24 descartados: los 20 pisos (17-59 rivales),
    plastic_sheeting#207 (2), glass_door#118 (3), black_door_frame#192 (5),
    white_wall#101 (6).

Cruzado con el acuerdo de forma (pico de correlación e IoU después de alinear),
los dos que pasan las dos cosas son `desk#203` (pico 0,86 · IoU 0,58 · ambig 0)
y `glass_door#117` (0,76 · 0,54 · 1) — **los dos que el usuario eligió a mano**
para su época 1 y su época 3.

### Código

```python
def ambiguity(copy_a, oid, label, points_by_oid, label_of, xyz, magnitude_m):
    tree = cKDTree(_sub(copy_a))
    hits = []
    for other, idx in points_by_oid.items():
        if other == oid or label_of.get(other) != label:
            continue
        dd, _ = tree.query(_sub(xyz[idx]), k=1)
        if dd.min() <= magnitude_m:
            hits.append(other)
    return len(hits), sorted(hits)

# y el filtro, con |t| ya medido:
kept = [(c, dr) for c, dr in pairs
        if ambiguity(c.copies[0], c.oid, c.label, points_by_oid,
                     label_of, xyz, dr.magnitude)[0] < max_ambiguity]
```


---

## Paso 12 — el filtrado real de la nube

`correction/visit_drift.py::cloud_filter_masklets`

Va **después** de corregir la pose y corre **por época**: cada nube tiene sus
objetos filtrados y no tienen por qué ser los mismos (USER 2026-09-18).

Tres reglas, todas sobre los **masklets**:

1. un punto que **dice ser parte de un objeto** y, ya corregida la pose, sigue
   cayendo **fuera de la máscara de ese objeto** en todas las vistas que lo
   vieron, no es parte de él
2. un masklet con menos de `min_points` (500) no se puede medir y no retiene nada
3. una visita que aporta `min_visit_share` (25 %) o menos de su masklet vio un
   PEDAZO de él y
   no retiene nada

**Los puntos no segmentados NO se tocan** (USER: *"no me elimines lo
unsegmented"*). Un punto que no pertenece a ningún masklet no tiene máscara que
lo juzgue, y el silencio no es un veredicto. En pccr son 5,55 M de puntos, un
cuarto de la nube.

Los masklets se solapan, así que **un punto sobrevive si algún masklet lo
retiene**: cae fuera en uno y bien en otro.

### Cómo se juzga: la vista CRUZADA

- **La vista propia no vota.** El punto se mueve junto con la cámara donde
  nació, así que su proyección ahí es idéntica antes y después de corregir. No
  dice nada. Se juzga en los keyframes de las **otras** visitas.
- **Una vista vota sólo si vio el punto**: dentro del frustum y sin geometría
  medida delante (z-buffer del propio keyframe, tolerancia
  `segmentation.mask_filter.depth_tol_m`).
- Se borra el punto **juzgado que nunca cayó adentro**, con la máscara dilatada
  `dilate_px` — las siluetas de SAM3 no son exactas al píxel y la nube es rala a
  resolución de máscara.
- Los z-buffers cacheados siguen valiendo después de corregir: los puntos de un
  keyframe y su cámara se mueven con la misma transformación, así que lo que esa
  cámara mide de sí misma no cambia. Lo que sí hay que actualizar son las poses
  de proyección.

### Medido sobre pccr

    22.120.080 puntos
    se van 311.250  (1,4 %)
      480.530 siguen fuera de su propia máscara tras corregir
            3 masklets con menos de 500 puntos
           45 visitas que aportan <= 1 %
    5.552.521 no segmentados, sin tocar

## Paso 13 — abierto

*"por ahora dejalo abierto, vamos a ejecutar siempre hasta epoch 0 (original),
1, 2, 3 y 4, forzosamente siempre"* (USER 2026-09-18). `max_epochs: 4` es una
cuenta, no un tope de seguridad. La única salida temprana que queda es que una
pasada mida una corrección por debajo de lo que la sesión puede repetir: ahí no
publica, porque escribir ruido no es converger.


---

## Paso 0 — el cierre de lazo por chunks

`correction/floor_consensus.py::measure_chunks` + `visit_drift_run.py::chunk_floor_epoch`

Corre **primero**, antes de la cadena de objetos. USER 2026-09-19: *"debe ser
parte del pipeline … antes prácticamente de empezar la corrección, porque ya te
deja muy bien acotado los errores"*.

Es el mismo discriminador de todo el módulo —**lugar contra momento**— pero
entre CHUNKS, que es donde mejor condicionado está: siete incógnitas contra
miles de celdas de piso que dos chunks vieron los dos.

### Lo que NO hay que hacer, y se midió

Nivelar cada chunk a y=0 **por su propia altura media** lo mueve por el relieve
real del edificio (306 a 632 mm en pccr) y, como los chunks se solapan a la
mitad, parte la zona compartida en dos alturas: **crea duplicado vertical donde
no lo había**, y estira todo objeto que cruce una costura. Probado y descartado.

### Lo que funciona

Comparar **sólo las celdas que los dos chunks vieron**. El relieve se cancela
porque los dos están mirando los mismos lugares:

    par de chunks   celdas compartidas   diferencia de altura
      0 <-> 6              1.118              +247,8 mm    <- el recorrido cerrando sobre sí mismo
      1 <-> 2                381               +50,9 mm
      2 <-> 3                541               +22,1 mm
      3 <-> 4                 61                +8,1 mm
      4 <-> 5                 53                −8,0 mm
      5 <-> 6                288               −12,0 mm

Chunks consecutivos difieren de 8 a 51 mm; el arranque y el final, que ven el
mismo piso porque la caminata vuelve, difieren **248 mm**. Ésa es la deriva
vertical acumulada.

Se resuelve `c_i − c_j = d_ij` por mínimos cuadrados ponderados por celdas
compartidas, con calibre de media cero para no levantar la escena entera. En
pccr el grafo es un árbol y **cada arista queda satisfecha con residuo 0,0 mm**.

Los desplazamientos se aplican con **mezcla lineal en cada costura**, sobre el
solape que el propio plan de chunks declara, así que nada se corta: el salto
máximo entre keyframes vecinos quedó en **6,2 mm**.

### Resultado en pccr

    corrección máxima   208,6 mm (el chunk 0, el arranque)
    salto entre kf      6,2 mm
    residuo             0,0 mm en las seis aristas
    puntos borrados     ninguno

Veredicto del usuario: *"el piso quedó nivelado perfecto en altura, y se
respetó la rampa que tenía"*.


## Paso 0b — la corrección de PROFUNDIDAD (USER 2026-09-19)

**Código:** `correction/visit_drift.scale_rows()` (la lectura),
`correction/visit_drift_run.scale_epoch()` (la época),
`reconstruction/certify/scale_stage.py` (el grafo, ya existía).

Los duplicados de pccr están separados **a lo largo de la línea de visión**, no
de costado: cinco de seis cierres son 97-99 % **radiales** (`desk#203`: 67,2 cm
de 67,8). Una traslación es la misma en todos lados; un error de **profundidad**
crece con la distancia. Por eso el desk cerraba a 0,7 cm a 3,6 m mientras las
líneas de baldosa a 8 m seguían 18,1 cm corridas, y por eso ningún par de
objetos coincidía nunca en un vector: cada uno está en un **rumbo** distinto, así
que UN error de profundidad se vuelve un vector del mundo distinto para cada uno.

Dos instrumentos independientes coinciden en el tamaño: las siluetas SAM3
(`k(d) = 1 + eps·d` ajusta los cierres con UNA incógnita mejor que la traslación
por keyframe con tres, rms 16,7 contra 22,8 cm) y las anclas DA3 de
`scale_diagnostics.json`, que no ven una silueta en su vida (escala por frame
0,8784 primera mitad → 1,0015 segunda, cociente **1,1401**, IC 90 %
[1,008, 1,256], Spearman rho +0,459 p 0,0038 sobre 38 frames).

**Cada cierre como razón de profundidad**, con la visita 1 de gauge:

    k_b = 1 + (t · u) / D_b              s_ab = 1 / k_b

**Nada se veta.** La parte **tangencial** —la que el modelo de profundidad no
puede producir— entra como residuo de la fila, así un cierre de costado ensancha
su propia barra de error en vez de ser rechazado por un umbral que nadie midió.
`glass_door#121`, la única identidad falsa de pccr (un TRIPLICADO: tres puertas
paralelas en un masklet), es 91 % tangencial y se descontó sola, σ 0,267 contra
0,054 del desk.

Las filas se publican en `scale_loop_rows.json` **selladas con la época en que se
midieron**: esta corrección cambia las profundidades mismas de las que están
hechas las filas, así que `scale_epoch` re-mide cuando el sello está viejo en vez
de componer una corrección ya aplicada. Se le suma la **forma** de la deriva DA3
como filas **relativas entre chunks consecutivos** (nunca absolutas: una fila
relativa no opina sobre el metro, sólo sobre cómo se reparte), con lo ya aplicado
restado vía los acuerdos relativos al lock.

**Se aplica por CHUNK**, no por frame: cada chunk trae su propio gauge, y un
cambio de profundidad por frame rompería la consistencia multi-vista que la
reconstrucción todavía tiene DENTRO de un chunk.

Resultado en pccr, época 1: r 0,9586→1,0432, las cámaras se corren ≤31,3 cm y la
nube ≤66,9 cm —el orden de la separación de los duplicados, alcanzado **sin medir
una sola traslación**—. El desk pasó de 67,8 a **11,2 cm** y los cierres dejaron
de ser radiales (64-66 %): lo que queda es tangencial, que es traslación y es el
trabajo de los pasos 1 a 11. **El orden correcto es profundidad primero.**

**LÍMITE DECLARADO:** las anclas muestran deriva **continua** y siete chunks sólo
pueden escribir una **escalera**. Con `sigma_seam_log` 0,02 y seis costuras el
modelo topa cerca del 12 %; pccr necesita ~14 %. No se arregla retocando el 0,02
—los frames de solape sí miden la escala relativa al 0,1-1 %—: el techo es
estructural, y el arreglo de fondo es que `scale_align` aprenda un modo a lo
largo del **recorrido**, el eje que su A/B de 2026-08-11 nunca probó (probó
**profundidad**, `s·z+b` y `a0·z+a1·z²`, y con razón no encontró estructura ahí).

## El consenso de piso POR KEYFRAME: eliminado (USER 2026-09-19)

*"eliminalo, ese paso ya no hace falta verdad?"*

Resolvía `h = T[celda] + dz[keyframe]` por medianas alternadas y movía cada
keyframe por su propio `dz`. Funcionaba sobre el piso —el `dz span` de pccr
bajó de 157,9 a 26,4 mm— **y rompía los objetos**: empujó la primera visita del
desk 32,8 mm abajo y la segunda 30,0 mm arriba, abriendo 62,8 mm un objeto que
estaba cerrado, todo en vertical.

La razón es estructural y ya la conocíamos: el piso y los objetos restringen
**los mismos keyframes**, y aplicar uno después del otro siempre rompe al
segundo.

El **paso 0** hace el mismo trabajo donde está bien condicionado —siete
incógnitas en vez de 216— y deja los objetos en paz. Con él andando, éste no
hacía falta.

Se fueron con él: `FloorSolution`, `solve()`, `measure()`, y los parámetros
`max_iters` y `chunks_enabled` (ahora `enabled` gobierna el paso 0). Quedan
`floor_points`, `observe`, `chunk_of_keyframe`, `chunk_edges`,
`solve_chunk_offsets` y `measure_chunks`.
