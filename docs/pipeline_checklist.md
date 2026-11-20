# The pipeline checklist — the single list

USER 2026-11-20: *"hay que armar una lista en la que este volcado esto, con las conclusiones de lo que esta bien, lo que
hay que corregir, corregir es eliminar, o hacer bien, etc"* / *"guardalo y es lo que vas actualizando, en la medida que
vas avanzando"* / *"la lista no debe ser una ensalada de pendientes, debe ser exactamente la hoja de ruta de lo que se
va a hacer"*.

ONE list: the 36 conclusions of the audit (`docs/pipeline_audit_2026-11-20.md`), one per line, never merged. Each line
says what the conclusion is, what has to be done with it, and its state. "C" is the conclusion's own number: it never
changes. "Order" is the order of execution of what is pending: the list is the roadmap, executed from 1 on, one line at
a time, and no line is started without his order.

What has to be done with a conclusion:
- **NOT TO BE TOUCHED** — it is right and necessary as it is.
- **REMOVE** — delete it from the code and refactor the code around it.
- **DO IT RIGHT** — it is right in concept and wrong as applied: correct it.
- **BUILD** — it does not exist yet.
- **MEASURE, THEN DECIDE** — the audit could not judge it: measure it on one session and he decides.

**THE RULE OF "DONE"** (USER 2026-11-20): *"cuando algo esta hecho, solo se considera hecho si el pipeline lo ejecuta,
si esta en otro lado, escrito tirado en un costado, en test, eso no sirve, solo se considera hecho si en el pipeline al
ejecutar la reconstrucion se ejecuta ese codigo, cuando algo debe eliminarse es eliminarse por completo, no un flag
apagado en el config, es eliminar del codigo, refactorizar el codigo"*.
- Built or corrected: DONE only when the reconstruction pipeline EXECUTES that code. Written on the side, in a scratch
  folder or only in a test: not done.
- Removed: DONE only when it is DELETED from the code and the code refactored. A switch turned off is not a removal.
- A line is DONE or PENDING. Nothing is "started".

State on 2026-11-20: 36 conclusions — 8 not to be touched, 1 done, 27 pending.

## The list

| Order | C | Conclusion | What has to be done | State |
|---|---|---|---|---|
| 1 | 34 | PointDiT (the detail at the edges): a model under a non-commercial licence, an hour of the card, no result or need shown — *"hay que sacarlo, eliminalo de la cadena, del repo, de todo"* | REMOVE — from F6's code, the configuration, the tests, the pod's environment and weights, the rebuild kit, the documents | PENDING |
| 2 | 16 | F6's depth factor per keyframe, and its "verification" on F5's own landmarks | DO IT RIGHT — one factor for the session; the run on the copy of pccr 2026-08-24 for his eye | PENDING |
| 3 | 31 | Every step is judged against itself; none against an independent reference | BUILD — the decision by zone at the end of every stage, the best state kept (the flow below), with one measure computed on the card | PENDING |
| 4 | 9 | F5 does not carry the cube graph: where the tracks say nothing it moves the cameras freely | DO IT RIGHT — relative constraints between neighbouring cameras and the closures inside F5's cost; the independent position priors deleted | PENDING |
| 5 | 15 | The old F2 (a scale along the walk) | REMOVE — today it is only switched (`gauge.profile: constant`) | PENDING |
| 6 | 18 | Today's F2 (a second estimate of the metre) and the DA3 windows it generates again | REMOVE | PENDING |
| 7 | 27 | The chain, the walk length and the chainage of the intake's DA3 windows | REMOVE | PENDING |
| 8 | 10 | The scale reaches the cameras and never the depth | DO IT RIGHT — the scale decided once and carried to cameras and depth together | PENDING |
| 9 | 24 | The fit of F6's session factor (the factor is known) | REMOVE — F6 multiplies by the known factor | PENDING |
| 10 | 20 | The DA3 lock per cube | REMOVE | PENDING |
| 11 | 13 | The 15 m range is evaluated in each cube's own units, which cannot be trusted | DO IT RIGHT | PENDING |
| 12 | 17 | The depth graph | REMOVE — today it is only switched off (`depth_graph: false`) | PENDING |
| 13 | 21 | The co-visibility budget of the cut | REMOVE | PENDING |
| 14 | 32 | The stronger verifications the code already has (a closure's split halves, the graph's leave-one-out) are not used on cubes | DO IT RIGHT — the cube stage executes them; a solve that does not converge stops the run | PENDING |
| 15 | 11 | The closures between passes: a weight that does not measure what they assert, never verified, selected on any overlap | DO IT RIGHT — weighted and selected by their verification | PENDING |
| 16 | 26 | The bridges over 6 m | REMOVE — a bridge that fails its verification is not used | PENDING |
| 17 | 23 | F4's witness windows and F5's localisation of the witness frames | REMOVE — with the intake's witness pass | PENDING |
| 18 | 12 | F4: three loop windows in four link no two passes | DO IT RIGHT — a loop window only where it links two passes | PENDING |
| 19 | 14 | The brightness gate of the intake | REMOVE | PENDING |
| 20 | 19 | The coverage fill | REMOVE | PENDING |
| 21 | 25 | The cube check | REMOVE | PENDING |
| 22 | 28 | The second copy of the cloud | REMOVE | PENDING |
| 23 | 29 | The two viewer layers | REMOVE — with the route that serves them | PENDING |
| 24 | 30 | The files nothing reads (the per-frame uncertainty maps are already deleted, fork bc93d56) | REMOVE | PENDING |
| 25 | 33 | The second pass of every cube and the turn frames | MEASURE, THEN DECIDE | PENDING |
| 26 | 35 | The vote's limit of 30°, and "in place" re-admitting what the majority or the veto rejected | MEASURE, THEN DECIDE | PENDING |
| 27 | 36 | F5's two focal lengths: whether they should be one | MEASURE, THEN DECIDE | PENDING |
| — | 22 | The seams' scale bound | REMOVE | **DONE** (2026-11-20: deleted from the fork's code, bc93d56 / 50fc503) |
| — | 1 | The frames, and never more than 8 video frames without a keyframe | NOT TO BE TOUCHED | OK |
| — | 2 | The first pass (who sees whom) | NOT TO BE TOUCHED | OK |
| — | 3 | The cut into cubes by his numbers, and the owner of each keyframe | NOT TO BE TOUCHED | OK |
| — | 4 | Each cube's pass, the seams, the closures between passes and the graph | NOT TO BE TOUCHED | OK |
| — | 5 | The one scale (scale_align) and the first orientation | NOT TO BE TOUCHED | OK |
| — | 6 | The first camera (F0), F4's tracks and F5's adjustment | NOT TO BE TOUCHED | OK |
| — | 7 | F6's vote and the cleaner | NOT TO BE TOUCHED | OK |
| — | 8 | The publication and the floor's levelling | NOT TO BE TOUCHED | OK |

## The flow agreed on 2026-11-20 (what line 3 builds)

*"si empeora ese paso o esa decisión se vuelve al estado mejor y se sigue con el siguiente paso del proceso. Debemos
llegar a la publicación de la nube habiendo pasado por todas las etapas y habiéndonos quedado con el mejor estado del
sistema"* / *"Se entiende perfecto y es la manera correcta de abordarlo"*.

1. There is always a BEST STATE: cameras + depth + scale.
2. A stage starts from the best state and solves the WHOLE session: a candidate.
3. The candidate is measured ZONE BY ZONE against the best state, with one measure that does not depend on the stage
   (how well different passes agree over the same surface, and how well neighbouring keyframes agree, in the images).
   A zone is a stretch of the walk (a cube's keyframes). The rule is his: 95 % and 1.1 × the measured error.
4. No zone worse → the candidate is the new best state.
5. Some zone worse → the same stage is solved AGAIN with those zones held to the previous state, and measured again.
6. Still worse somewhere → the stage is discarded, the best state stays, the reason is recorded.
7. Never cut and paste two solutions: always ONE coherent state. After every stage no zone is worse than before.
8. Decided on the fly, unit by unit, only what is independent from one keyframe to the next (the vote, pixel by
   pixel).
9. The cloud that is published is the best state's, with the record of every decision.

## Besides the 36 (not conclusions of the audit; kept apart)

- The rebuild kit of the pod captured again after the fork's commit bc93d56 — PENDING.
- A run cannot be lost at its last step: F6 keeps what it computed until its cloud is published; the cleaner takes a
  scratch on the volume when the local disk is short — PENDING.
- `correction/release.py`: its two log divisors as integers again, once no chain waits on its old bytes — PENDING.
- What still runs on the CPU in a stage moves to the card when that stage is touched — in every line above.
