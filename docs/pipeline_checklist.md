# The pipeline checklist — the roadmap, in the exact order of execution

USER 2026-11-20: *"hace un checklist con todas las conclusiones, lo que esta OK, marcalo como OK no se toca, lo que hay
que hacer y ya esta hecho con ok, realizado, y lo que hay que hacer y no esta hecho marcalo como pendiente"* / *"no me
lo pongas a mi aca, guardalo y es lo que vas actualizando, en la medida que vas avanzando"* / *"la lista no debe ser una
ensalada de pendientes, debe ser exactamente la hoja de ruta de lo que se va a hacer"*.

This file is the single list. It is updated every time a step closes. The steps are executed in this order, one at a
time; a step is not started without his order. "C-n" is conclusion n of the audit
(`docs/pipeline_audit_2026-11-20.md`; the 36 conclusions are listed at the end of this file).

**THE RULE OF "DONE"** (USER 2026-11-20): *"cuando algo esta hecho, solo se considera hecho si el pipeline lo ejecuta,
si esta en otro lado, escrito tirado en un costado, en test, eso no sirve, solo se considera hecho si en el pipeline al
ejecutar la reconstrucion se ejecuta ese codigo, cuando algo debe eliminarse es eliminarse por completo, no un flag
apagado en el config, es eliminar del codigo, refactorizar el codigo"*.
- Something built or corrected is DONE only when the reconstruction pipeline EXECUTES that code.
- Something to remove is DONE only when it is DELETED from the code and the code is refactored around it. A switch
  turned off in the configuration is not a removal.
- A step is DONE or PENDING. Nothing is "started".

State on 2026-11-20: 16 steps, 0 done, 16 pending.

## The roadmap

| Step | What is done in it | Closes | It is DONE when | State |
|---|---|---|---|---|
| 1 | The rebuild kit of the pod captured again (owed since the fork's commit bc93d56) | — | `drift.py` says the kit is the pod; committed and pushed | PENDING |
| 2 | A run can no longer be lost at its last step: F6 keeps what it computed until its cloud is published and resumes AT the publication; the cleaner takes a scratch on the volume, declared, when the local disk is short | — | the pipeline's F6 and cleaner are that code; a failed publication resumes without computing again (tested) | PENDING |
| 3 | **PointDiT deleted** — from F6's code, the configuration, the tests, its environment and weights on the pod, the rebuild kit, the documents (his order: *"hay que sacarlo, eliminalo de la cadena, del repo, de todo"*) | C-34 | F6 runs and no PointDiT code, key, weight or mention is left | PENDING |
| 4 | F6 on the copy of pccr 2026-08-24 with ONE depth factor for the session — the run he is waiting for. After it, `correction/release.py` gets its two divisors back as integers | C-16 (his eye) | the cloud is published and he has looked at it | PENDING |
| 5 | The coherence measure inside the pipeline: computed on the card after the cube graph and after F5, written with the session's products | C-31 (the measure) | a reconstruction run executes it and its numbers are in the session | PENDING |
| 6 | F5 carries the cube graph: relative constraints between neighbouring cameras with the cubes' own uncertainty, and the closures, inside F5's cost; the independent position priors deleted. bufferStop first, then the copy | C-9 | the pipeline's F5 is that code; the measure and his eye on one session | PENDING |
| 7 | The decision by zone after F5: no zone worse → the result is the best state; zones worse → solved again with those zones held to the previous state; still worse → the stage is discarded. The decision is written with the products | C-31 | the pipeline executes the decision after F5 and records it | PENDING |
| 8 | F2 deleted: the scale along the walk, today's second estimate of the metre, the DA3 windows it generates again, and the intake's chain / walk length / chainage that only fed it | C-15, C-18, C-27 | no F2 step in the chain and no code of it left | PENDING |
| 9 | The scale decided once and carried to the depth: F6 multiplies by the known factor; the code that fitted and "verified" a depth factor deleted | C-10, C-16, C-24 | the pipeline's F6 has no fitted factor and no code of it left | PENDING |
| 10 | The cube stage: the DA3 lock per cube deleted and the 15 m range evaluated in units that can be trusted; the depth graph's code deleted; the co-visibility budget of the cut deleted | C-13, C-20, C-17, C-21 | the cube stage runs without that code existing | PENDING |
| 11 | The closures between passes: verified by what they assert (the split-half test the code already has), weighted and selected by it; the graph judged (leave-one-out); a solve that does not converge stops the run. A bridge that fails its verification is not used | C-11, C-32, C-26 | the cube stage executes the verification and the judge | PENDING |
| 12 | The decision by zone after the cube graph | C-31 | the pipeline executes the decision after the graph and records it | PENDING |
| 13 | F4: the witness windows, F5's localisation of the witness frames and the intake's witness pass deleted; a loop window only where it links two passes | C-23, C-12 | F4 and F5 run without that code existing; loop windows are chosen by what they link | PENDING |
| 14 | The intake: the brightness gate deleted; the coverage fill deleted | C-14, C-19 | the intake runs without that code existing | PENDING |
| 15 | The publication: the cube check deleted; the second copy of the cloud deleted; the two viewer layers and their route deleted; the files nothing reads deleted | C-25, C-28, C-29, C-30 | the chain ends at the publication and none of those is written or served | PENDING |
| 16 | To measure, then decide with him: whether the second pass of every cube adds anything; the vote's limit of 30° and "in place"; whether F5's two focal lengths should be one | C-33, C-35, C-36 | each is measured on one session and decided by him | PENDING |

In every step: what still runs on the CPU in the stage that is touched moves to the card (*"todo en placa nada en
cpu"*), and a change that alters what he sees is shown to him on ONE session before it reaches the others.

Already deleted from the code before this roadmap (2026-11-20, fork bc93d56): the seams' scale bound (C-22) and the
per-frame uncertainty maps (a part of C-30). Only switched off, NOT done: the depth graph (`depth_graph: false`) and
the old F2 (`gauge.profile: constant`) — steps 10 and 8 delete them.

## The 36 conclusions of the audit (their numbers never change)

Right and necessary — NOT TO BE TOUCHED:
1. The frames, and never more than 8 video frames without a keyframe.
2. The first pass (who sees whom).
3. The cut into cubes by his numbers, and the owner of each keyframe.
4. Each cube's pass, the seams, the closures between passes and the graph.
5. The one scale (scale_align) and the first orientation.
6. The first camera (F0), F4's tracks and F5's adjustment.
7. F6's vote and the cleaner.
8. The publication and the floor's levelling.

Right in concept, wrong as applied — to correct:
9. F5 does not carry the cube graph. → step 6
10. The scale reaches the cameras and never the depth. → step 9
11. The closures: a weight that does not measure what they assert, never verified, selected on any overlap. → step 11
12. F4: three loop windows in four link no two passes. → step 13
13. The 15 m range is evaluated in each cube's own units. → step 10

Wrong by concept — to remove or replace:
14. The brightness gate of the intake. → step 14
15. The old F2. → step 8
16. F6's depth factor per keyframe and its "verification" on F5's own landmarks. → steps 4 and 9
17. The depth graph. → step 10

Redundant — to remove:
18. Today's F2 and the DA3 windows it generates again. → step 8
19. The coverage fill. → step 14
20. The DA3 lock per cube. → step 10
21. The co-visibility budget of the cut. → step 10
22. The seams' scale bound. → deleted (fork bc93d56)
23. F4's witness windows and F5's localisation of the witness frames. → step 13
24. The fit of F6's session factor (the factor is known). → step 9
25. The cube check. → step 15
26. The bridges over 6 m. → step 11
27. The chain, the walk length and the chainage of the intake's DA3 windows. → step 8

Duplicated, or written and not used — to remove:
28. The second copy of the cloud. → step 15
29. The two viewer layers. → step 15
30. The per-frame uncertainty maps (deleted, fork bc93d56) and the other files nothing reads. → step 15

The verification:
31. Every step is judged against itself; none against an independent reference. → steps 5, 7, 12
32. The stronger verifications the code already has (closures, graph) are not used on cubes. → step 11

Not judged by the audit:
33. The second pass of every cube and the turn frames. → step 16
34. PointDiT — his decision of 2026-11-20: deleted. → step 3
35. The vote's limit of 30° and "in place". → step 16
36. F5's two focal lengths. → step 16
