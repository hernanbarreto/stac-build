# MEJORAS OBLIGATORIAS — deuda que bloquea el trabajo diario

Estado: PENDIENTE. Orden del usuario (2026-09-06): "obligatorio todo".
Nada de esto es opcional; cada punto se implementa, se mide y se cierra.
Números medidos en pccr_v1 (122,7 M puntos, 1329 keyframes, A6000).

---

## 1. Archivo de segmentación gigante (785 MB) — cuelga la carga

**Problema.** `segmentation_result.json` guarda los índices de CADA punto de
cada instancia. Crece con los puntos segmentados, no con los objetos:
2 objetos → 31 MB; 31 objetos → **785 MB**. Cargar la sesión parsea todo eso
(GBs de RAM, minutos) y se lo manda entero al viewer, que nunca lo usa.
`scene_r.db` tiene el mismo defecto (935 MB).

**Criterio de cierre.** 31 objetos → JSON < 100 KB; carga de sesión < 2 s;
ledger del brush balanceado; octree verificado ≥ 0,995.

**Diseño completo (aprobar antes de implementar):**


### 1.1 Problem (measured on pccr_v1)

`segmentation_result.json` stores, for every instance, the full list of cloud
point indices (`globalIndices`). Its size therefore grows with the number of
SEGMENTED POINTS, not with the number of objects:

| instances | segmented points | file size |
|---|---|---|
| 2 (door1, rack1) | 3.2 M | 31 MB |
| 31 | 81.6 M (66.5 % of the cloud) | **785 MB** |

Consequences:
- Session load parses 785 MB of JSON (`json.load` → several GB of RAM, minutes,
  CPU-bound) — the backend appears hung.
- The whole object is then sent to the viewer over the WebSocket / the
  `GET /api/sessions/{id}/segmentation` reply.
- **The frontend never reads `globalIndices`** (0 references in `ui/src`).
- `scene_r.db` (instance store) also ballooned to 935 MB (separate follow-up,
  §1.9).

The per-point membership ALREADY exists in a compact binary form:
`classification.npy` — one instance id per cloud point (117 MB, uint8, 0 =
unsegmented). It is what the Potree converter bakes into the octree
(`potree_converter.py:106`), what the brush rebuilds after every stroke
(`erase.py:_write_classification`), and what the octree verification samples.

### 1.2 Principle

**One source of truth for point membership: `classification.npy`.**
`segmentation_result.json` keeps only per-instance METADATA. Point indices are
DERIVED on demand: `np.flatnonzero(classification == instance_id)`.

This also satisfies the user's doctrine (2026-09-06): every mutation leaves
the disk consistent; a load reads and shows, never recomputes.

### 1.3 New on-disk contract

`segmentation_result.json` v4 (constant size, ~1 KB per instance):

```json
{
  "type": "segmentation", "version": "4.0",
  "membership": "classification.npy",
  "total_points": 122752266, "segmented_points": 81621594, "coverage": 0.665,
  "resolution": {"original": [832, 464], "scaled": [832, 464]},
  "instances": [
    {"instance_id": 1, "id": 0, "label": "door1", "color": "#FFFF00",
     "total_points": 163236, "obb": {...}, "propagated": true}
  ]
}
```
No `globalIndices`. `total_points` per instance = `bincount(classification)`.

`classification.npy`: unchanged format (per-point instance id, same order as
`cleaned_cloud.ply` / `corrected_cloud.ply` / the Potree LAS — this ordering
assumption is already relied on everywhere: `cloudcompy_worker.py:201,234`,
`surface_fit/consolidate.py:534`, `tsdf_export.py:3901`).

### 1.4 Access layer — `server/segmentation/membership.py` (new)

The ONLY module that reads or writes point membership.

```python
load_classification(output_dir) -> np.ndarray          # np.load(mmap_mode="r"), cached per (path, mtime)
instance_indices(output_dir, iid) -> np.ndarray[int64]  # flatnonzero(cls == iid)
instance_indices_many(output_dir, iids) -> dict         # ONE pass (argsort/bincount) for callers needing many instances
instance_counts(output_dir) -> dict[iid, n]             # bincount
# writers (atomic: np.save to tmp + os.replace)
assign_points(output_dir, idx, to_iid)                  # brush reassign; to_iid=0 → unsegmented
delete_instance(output_dir, iid)                        # cls[cls == iid] = 0
remap_after_physical_delete(output_dir, keep_mask)      # cls = cls[keep]  (replaces erase.py:655-670)
rewrite_from_masks(output_dir, cls_array)               # matching writes the whole array
# compatibility shim — the migration hinge
inst_indices(inst, output_dir) -> np.ndarray[int64]
    # v3 file (has "globalIndices") → uses them; v4 → instance_indices(...)
```

Cost on the 122.7 M-point cloud: `cls == iid` ≈ 60 ms (uint8 compare),
bincount ≈ 100 ms, mmap avoids re-reading 117 MB. Callers that need every
instance (surface_fit scene, BIM comparison, store rebuild) use
`instance_indices_many` — a single argsort (~1–2 s) instead of N scans.

### 1.5 Migration map (from the inventory, `grep globalIndices`)

#### Writers (stop emitting `globalIndices`)
| site | change |
|---|---|
| `segmentation/pipeline.py` `_match_masks_to_cloud` (~1998-2040, 2285) | already builds `classification`; drop `globalIndices` from the instance dict, keep `total_points` (bincount) + `obb` |
| `pipeline.py` merge/dedupe (1605-1665) | operate on classification (`assign_points`) |
| `segmentation/erase.py` (20 sites) | mutate classification directly: reassign → `assign_points`, delete-mode → `assign_points(idx, 0)`, physical delete → `remap_after_physical_delete(keep)`; `_write_classification` disappears (classification IS the state); the ledger/verify code keeps working (it already reads classification) |
| `correction_analysis._update_result_obbs` | `instance_indices` instead of `inst["globalIndices"]` |
| `main.py` level_floor `_recompute_result_obbs` (2432-2544) | same |
| `main.py` rename/delete/clean_instance/paint_mask | delete → `delete_instance`; others unaffected (metadata) |
| `propagation_resume.cancel` | already `cls[cls == iid] = 0` ✓ |
| `pipeline.rebuild_instance_store` (2436) | from `instance_indices_many` |

#### Readers (replace one identical line, ~45 sites, 20 files)
`np.asarray(inst.get("globalIndices") or [], dtype=np.int64)` →
`inst_indices(inst, output_dir)`:
`segmentation/{tsdf_export,poisson_object,perfect_object,mesh_export,object_analysis,shape_proposer}.py`,
`reconstruction/surface_fit/{scene,runner}.py`, `bim/{registration,comparison,occlusion_raycaster}.py`
(these use `inst.get("globalIndices", inst.get("point_indices"))` — same shim),
`run_pgsr_object.py`, `reconstruction_runner.py`. Mechanical; the shim keeps
v3 files working during the transition.

#### API / viewer
- `GET /api/sessions/{id}/segmentation` (main.py 3039+) and the load-time
  WebSocket broadcast (`apply_segmentation_to_cloud` → `send_text`) send
  instances WITHOUT indices (the UI never used them) → payload in KB.
- `apply_segmentation_to_cloud` returns metadata only.

#### Tests
`server/tests/test_surface_fit_scene.py`, `test_mesh_export.py` build fixtures
with `globalIndices` → switch to a synthetic `classification.npy`.

### 1.6 Instance-id limit (hard, loud)

`classification.npy` is uint8 and the Potree LAS point format 7 classification
is uint8 (`potree_converter.py:84-116`) → **255 instances max**. Today the
code silently CLAMPS (`min(iid, 255)`, `erase.py:_write_classification`),
which corrupts membership past 255. v4: refuse to create instance 256 with a
clear error at the creation points (mask save, brush new-segment). Raising the
limit (uint16 + LAS extra dimension + viewer shader) is a SEPARATE decision.

### 1.7 Existing sessions (v3 files)

No automatic rewrite on load (doctrine: load never recomputes). Two pieces:
1. The `inst_indices` shim reads `globalIndices` when present → old sessions
   keep working untouched.
2. `scripts/migrate_segmentation_v4.py <project> [scan]` (explicit, run by the
   user): rebuilds `classification.npy` from the v3 indices if missing or
   inconsistent, rewrites the JSON as v4 without indices, keeps a `.v3.bak`
   until the user deletes it. Idempotent.

### 1.8 Consistency rules (doctrine, enforced)

- Every mutation (matching, brush stroke, resume, correction, floor leveling,
  cancel) ends with: classification.npy written atomically → JSON counts/OBBs
  rewritten → octree rebuilt/recolored only if classification changed
  (already conditional since fa65eb3). Load reads only.
- Writers hold the existing per-session matching lock (`_get_matching_lock`)
  while touching classification + JSON, so a brush stroke and a refresh cannot
  interleave.
- `verify_octree_classification` (erase.py) stays as the end-of-transaction
  check (agreement ≥ 0.995).

### 1.9 Out of scope (flagged, decide separately)
- `scene_r.db` at 935 MB: the instance store likely duplicates per-instance
  point data; it should hold OBB/stats/dossiers only and derive points via
  the access layer. Same principle, separate change.
- >255 instances (uint16 classification).

### 1.10 Rollout — three verifiable phases

**P1 — access layer + readers (zero behaviour change).** Add
`membership.py`; switch all readers to `inst_indices`. Verify on an existing
v3 session: meshing (poisson/tsdf), surface_fit, chat measurements, BIM
comparison give identical results (they still read the v3 indices).

**P2 — writers + API.** Matching/erase/correction/level_floor/resume stop
emitting indices; API/WS strip. Verify on pccr_v1 re-segmented from zero:
JSON < 100 KB with 31 objects; session load < 2 s; brush reassign/delete/
undo ledger balances; physical delete remaps; corrections update OBBs; octree
verify ≥ 0.995; all consumers of P1 still work on the v4 file.

**P3 — migration script + 255 guard + docs.** Run the migration on a copy of
an old session, compare counts/OBBs before/after (must be identical).

### 1.11 Risks
| risk | mitigation |
|---|---|
| a site keeps writing `globalIndices` into the JSON | P2 adds a write-time assertion in the single JSON writer: reject any instance carrying `globalIndices` |
| order mismatch cloud ↔ classification after a physical delete | one writer (`remap_after_physical_delete`) applied to cloud, corrected cloud and classification with the same keep mask |
| concurrent brush + refresh | matching lock around every write |
| performance on 122 M points | mmap + bincount/argsort helpers; measured budgets above |
| 255 clamp corrupting membership | hard refusal at creation |

---

## 2. TODO tarda demasiado — la GPU no es el cuello, es I/O + Potree

| operación | hoy | causa |
|---|---|---|
| Rebuild de Potree (tras corrección, brush, resume, undo) | **5–7 min** | PLY→LAS 5,3 GB en Python (~2,5 min) + PotreeConverter (~1 min) + copia de 5,7 GB (~30 s) |
| Brush, cada trazo | 1–5 min | reescribe `classification.npy` (122 M) + JSON de 785 MB + **reconstruye `scene_r.db`** + dispara un rebuild COMPLETO de Potree ("recolor" = rebuild) |
| Corrección (🔧 / gizmo / ⇩ piso) | ~7 min | parsea PLY 3,2 GB + reescribe + rebuild Potree |
| Cargar sesión | minutos | parsea 785 MB de JSON (punto 1) |
| Abrir Segmentation Manager | ~8 min | SAM3 decodifica 1329 JPEG a 2,5 fps |
| Matching máscaras→nube | 5–8 min | 122 M puntos contra máscaras, CPU |

### 2.1 Brush sin rebuild de Potree (5 min → segundos) — PRIORIDAD 1
El octree guarda la clasificación como un byte por punto en `octree.bin`.
Agregar `point_index` (uint32) como extra dim del LAS al convertir; un trazo
= **parchear esos bytes en el archivo** (una pasada de I/O, o solo los nodos
afectados) en vez de reconstruir 5,7 GB. El parser del octree ya existe
(`erase.verify_octree_classification`). El viewer recarga nodos.

### 2.2 Brush sin reescribir gigantes — PRIORIDAD 1
JSON a KB (punto 1) y `scene_r.db` reconstruido al cerrar el manager, no por
trazo.

### 2.3 Rebuild de Potree 3× más rápido cuando SÍ hace falta (geometría)
LAS escrito con numpy vectorizado; `mv` en vez de `cp -a` (mismo filesystem
→ instantáneo); sin cambio de encoding.

### 2.4 Correcciones sin parsear PLY
Sidecar `xyz.npy` (float32, 1,4 GB) + provenance en npy, memory-mapped:
leer 122 M puntos pasa de ~40 s a instantáneo. El PLY se reescribe solo al
aplicar.

### 2.5 SAM3: cachear frames decodificados por sesión
Segunda apertura del manager en segundos, no 8 min.

### 2.6 Matching en GPU
Proyección máscara→nube con torch (es álgebra por punto): minutos → segundos.

### 2.7 Estado "pendiente" ANTES del rebuild de Potree
Hoy el Undo recién aparece cuando el rebuild termina; si tarda o falla, no
hay Undo aunque el backup exista. Escribir el estado pendiente al aplicar
nube+poses, antes del Potree.

**Criterio de cierre.** Trazo de brush visible en < 5 s; corrección completa
< 2 min; abrir el manager la 2ª vez < 30 s; carga de sesión < 2 s.

---

## 3. Corrección automática: guardias que faltan (caso box1, 2026-09-06)

Con 464 puntos de un solo objeto a 7,3 m, el rígido devolvió **92° y 9 m**
(rms 0,9 cm — sobreajuste degenerado) y se aplicó pese a que el testigo del
piso empeoró de +0,1 a −22,7 cm. 5,4 M puntos y 60 cámaras rotos hasta el Undo.

1. **El gate held-out BLOQUEA**: si un testigo (piso u otro objeto) empeora
   más del umbral, la corrección se rechaza y NO se aplica.
2. **Tope de plausibilidad**: rot > 10° o |t| > 3 m ⇒ anclas rotas ⇒ rechazo
   (configurable; la deriva real medida es ~1–2°, ~1,5 m).
3. **Evidencia mínima**: un solo objeto con pocos puntos / disparo lejano ⇒
   solo traslación, o rechazo por evidencia insuficiente.
4. **El bbox no infla el objeto**: box1 curado = 37 k puntos, el OBB con
   margen tomó 282 k. Referencia = puntos del segmento curado; el bbox solo
   para encontrar copias, filtradas por cercanía a la superficie del segmento.

**Criterio de cierre.** Reproducir box1: la corrección se RECHAZA con el
motivo en pantalla; las correcciones válidas (rack ch29/30, piso) siguen
pasando.

---

## Orden de ejecución propuesto
1 → 2.1 + 2.2 (brush) → 3 → 2.3 + 2.4 (correcciones) → 2.7 → 2.5 → 2.6.
Cada punto con su medición antes/después en este archivo al cerrarlo.
