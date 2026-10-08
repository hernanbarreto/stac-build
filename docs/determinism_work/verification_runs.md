# Verificación final del determinismo (plan, "Verificación final"): dos corridas Omega → F6 sobre la misma sesión, todos los productos idénticos byte a byte

## pccr 2026-08-31 — par de corridas de la verificación

| | run2 (referencia) | run3 (verificación) |
|---|---|---|
| Lanzada | 2026-10-07 21:07 UTC (`logs/server_20261007_210725.log`) | en cola detrás de zaragoza, 2026-10-08 ~02:45 UTC |
| Código | main `1c466f4`, fork `67f30e9` (sin cambios de pipeline entre las dos; solo `ui/src/App.tsx`) | el mismo |
| Stamp del fork | `b1e3ccfef915…` (3342 inputs, 208 archivos de código) | debe repetirse |
| Hashes | `pccr_run2_20261007_2107_sha256.txt` (4080 archivos: frames, intake, output) | `pccr_run3_…_sha256.txt` |
| Copia completa | `/workspace/stac-keep/pccr_run2_20261007_2107/` (4.6 GB, verificada contra la lista) | — |

`pccr_run1_20261007_sha256.txt` es la corrida de la mañana (código anterior: sin keyframes de rotación,
plan sobre todos los keyframes); no es comparable con run2/run3.

Cómo se hizo la lista (desde el directorio del scan, rutas relativas, orden fijo):

    find . -type f -print0 | sort -z | xargs -0 sha256sum | sed -E 's#  \./#  #'

Comparación (cuando run3 termine):

    diff <(sort -k2 run2.txt) <(sort -k2 run3.txt)

Qué cuenta como diferencia ESPERADA (no son productos): `*timing*`, `output/potree/log.txt`,
`chain_state.json` si lleva fechas, y lo que escribe el VISOR al abrir la sesión (caché de
matching de máscaras, `floor_transform.npz`). Todo lo demás que difiera es una falla del determinismo
y se anota aquí con el archivo y la etapa que lo produce.

## zaragoza 2026-06-03

Corrida con el mismo código: lanzada 2026-10-07 23:57 UTC, stamp del fork `f1240c6be6b2…`
(185 inputs), un solo chunk (D_total 5.01). Sus hashes se guardan al terminar
(`zaragoza_run1_20261007_2357_sha256.txt`) para una verificación posterior.

## Resultado

(pendiente: se completa cuando run3 termine)
