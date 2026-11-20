# The pipeline checklist — the single list

USER 2026-11-20: *"hace un checklist con todas las conclusiones, lo que esta OK, marcalo como OK no se toca, lo que hay
que hacer y ya esta hecho con ok, realizado, y lo que hay que hacer y no esta hecho marcalo como pendiente"* / *"no me
lo pongas a mi aca, guardalo y es lo que vas actualizando, en la medida que vas avanzando"*.

The 36 conclusions of the audit (`docs/pipeline_audit_2026-11-20.md`). The numbers never change. Three states only:

- **OK — NOT TO BE TOUCHED**: right and necessary as it is.
- **OK — DONE**: had to be done and is done (tested, applied, committed).
- **PENDING**: has to be done and is not done. An item is done or it is not: nothing is "started".

Updated every time an item closes. Last update: 2026-11-20 — 8 not to be touched, 3 done, 25 pending.

## Right and necessary

| # | Conclusion | State |
|---|---|---|
| 1 | The frames, and never more than 8 video frames without a keyframe | OK — NOT TO BE TOUCHED |
| 2 | The first pass (who sees whom) | OK — NOT TO BE TOUCHED |
| 3 | The cut into cubes by his numbers, and the owner of each keyframe | OK — NOT TO BE TOUCHED |
| 4 | Each cube's pass, the seams, the closures between passes and the graph | OK — NOT TO BE TOUCHED |
| 5 | The one scale (scale_align) and the first orientation | OK — NOT TO BE TOUCHED |
| 6 | The first camera (F0), F4's tracks and F5's adjustment | OK — NOT TO BE TOUCHED |
| 7 | F6's vote and the cleaner | OK — NOT TO BE TOUCHED |
| 8 | The publication and the floor's levelling | OK — NOT TO BE TOUCHED |

## To correct (right in concept, wrong as applied)

| # | What has to be done | State |
|---|---|---|
| 9 | F5 carries the cube graph (where the tracks say nothing, the graph holds) | PENDING |
| 10 | The scale reaches the depth when it reaches the cameras | PENDING |
| 11 | The closures between passes: a weight that measures what they assert, their verification, their selection | PENDING |
| 12 | F4: a loop window only where it links two passes | PENDING |
| 13 | The 15 m range in units that can be trusted | PENDING |

## To remove or replace (wrong by concept, redundant, duplicated, unread)

| # | What has to be done | State |
|---|---|---|
| 14 | Remove the brightness gate of the intake | PENDING |
| 15 | Remove the old F2 (the scale along the walk) | OK — DONE (2026-11-20, one scale: `gauge.profile: constant`) |
| 16 | F6: one depth factor for the session instead of one per keyframe | PENDING (its test run on the copy of pccr 2026-08-24 did not finish) |
| 17 | Remove the depth graph | OK — DONE (2026-11-20, c97c36a: six real cubes, 276 products byte-identical without it) |
| 18 | Remove today's F2 and the DA3 windows it generates again | PENDING |
| 19 | Remove the coverage fill | PENDING |
| 20 | Remove the DA3 lock per cube | PENDING |
| 21 | Remove the co-visibility budget of the cut | PENDING |
| 22 | Remove the seams' scale bound | OK — DONE (2026-11-20, fork bc93d56, 50fc503) |
| 23 | Remove F4's witness windows and F5's localisation of the witness frames | PENDING |
| 24 | F6 uses the known factor instead of fitting it | PENDING |
| 25 | Remove the cube check | PENDING |
| 26 | The bridges over 6 m (his criterion first) | PENDING |
| 27 | Remove the chain, the walk length and the chainage of the intake's DA3 windows | PENDING |
| 28 | Remove the second copy of the cloud | PENDING |
| 29 | Remove the two viewer layers and their route | PENDING |
| 30 | Remove the files nothing reads (the per-frame uncertainty maps are gone: fork bc93d56; the others remain) | PENDING |

## To build

| # | What has to be done | State |
|---|---|---|
| 31 | The decision by zone at the end of every stage (the best state kept; the diagram agreed on 2026-11-20) | PENDING |
| 32 | The verifications the code already has for the closures and the graph, used on cubes | PENDING |

## To measure or decide

| # | What has to be done | State |
|---|---|---|
| 33 | Whether the second pass of every cube (the turn frames) adds anything | PENDING |
| 34 | **PointDiT: REMOVE IT from the chain, from the repo, from everything** — USER 2026-11-20: *"el detalle de bordes pointdit nunca me cerro un modelo con licencia no comercial que tarda un huevo sin un resultado ni necesidad aparente, hay que sacarlo, eliminalo de la cadena, del repo, de todo, no tiene ningun sentido"* | PENDING |
| 35 | The vote's upper limit of 30° and "in place" re-admitting what the majority or the veto rejected | PENDING |
| 36 | Whether F5's two focal lengths should be one | PENDING |

## Owed besides the audit (the incident of 2026-11-20; not conclusions — kept apart, with letters)

| | What has to be done | State |
|---|---|---|
| a | F6 keeps what it computed when its publication fails (a failed publication resumes at the publication) | PENDING |
| b | The cleaner takes a scratch on the volume when the local disk is short, instead of failing | PENDING |
| c | The rebuild kit captured again after the fork's commit bc93d56 | PENDING |
| d | `correction/release.py`: its two log divisors as integers again, once no chain is waiting on its old bytes | PENDING |
| e | What still runs on the CPU moves to the card in every stage that is touched | PENDING |
