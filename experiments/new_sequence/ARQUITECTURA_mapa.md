# Architecture guided by AMB3R's map — WORKING DRAFT (not coded, not run)

Working document of the design conversation of 2026-11-22 (USER: *"anda volcando todos los pasos de la arquitectura a
un documento para que lo actualicemos y cuando quede listo lo codificamos y probamos"*). The design text below is in
Spanish, as it is worked out with him; it is edited here as the conversation goes on. Nothing of it is coded: it is
coded and tested only when he says it is ready.

Background: `CHECKLIST_arquitectura.md` (the test of 2026-11-21 on three sessions, its numbers and his verdicts).

---

## El principio

- *"prefiero mantenerlo como lo tenemos ahora, con pasada amber y pasada omega, pero de forma mucho mas inteligente
  … sacarle todo el jugo a lo que ya tenemos de la pasada amber, para todas las etapas, seleccion de kf, seleccion de
  revisitas, seleccion de correspondencias con coherencia, atado … lo mismo el voto de f6 … todo debe estar
  respaldado por la nube y poses y campos de vision iniciales"*.
- *"no estamos ciegos, no hay que inventar nada, ya lo tenemos"*: la pasada de AMB3R ya dice qué frames miran qué
  cosa, dónde hay revisita y cómo está la cobertura de cada sector.

## Los pasos

Cada paso dice qué hace, qué lee y qué entrega. Lo que falta definir de cada uno está en "Por definir".

1. **Frames buenos.** ✓ CERRADO (2026-11-22). Un frame se descarta cuando:
   - su brillo medio pasa del máximo (quemado);
   - la parte de sus píxeles quemados (≥ 250) pasa del máximo (saturado);
   - su detalle es menos de la mitad del de sus vecinos (15 frames de cada lado): falta de nitidez o de textura.
   La oscuridad no descarta: un tramo oscuro se queda. Las perillas y sus valores, en "Perillas".
   Entrega: la lista de frames buenos. Un frame malo no puede ser keyframe ni testigo de F4. AMB3R sigue leyendo todos
   los frames (sigue la cámara frame a frame).
2. **AMB3R sobre todos los frames** (con Omega como modelo, como hoy).
   Entrega: las poses de AMB3R de todos los frames, la profundidad de AMB3R de uno de cada cuatro, y sus keyframes
   propios.
3. **Sonda DA3** sobre 16 keyframes propios de AMB3R, repartidos a lo largo del video.
   Entrega: el metro de AMB3R.
4. **El mapa.** Con las poses y la profundidad de AMB3R en metros, hasta 15 m: para cada zona, qué frames la ven,
   desde qué distancia y dirección, y en cuántas visitas del recorrido. Es la vista en planta: superficies, cámaras y
   campos de visión.
   Entrega: el mapa. Todos los pasos siguientes lo leen.
5. **Selección de keyframes**, sobre el mapa, solo entre frames buenos. Entra un keyframe nuevo cuando, respecto del
   último elegido, pasa alguna de estas tres cosas:
   - el solape de la vista baja de la franja de 92 a 96 %;
   - el desplazamiento de la cámara supera un ángulo;
   - la rotación de la cámara supera un ángulo.
   De los candidatos se elige el más nítido. No hay tope temporal ni relleno.
   Entrega: los keyframes, y los testigos de F4, que son los frames buenos que quedan entre keyframes.
6. **Cubos de Omega, por lugar.** Los keyframes se ordenan por el lugar que miran, no por el momento en que se
   tomaron, y se cortan en cubos de 64 keyframes compartiendo 32. Un lugar recorrido una vez da los cubos de hoy. Un
   lugar recorrido más de una vez tiene sus visitas dentro del mismo cubo, cuando el mapa dice que miran la misma
   superficie desde distancia y dirección compatibles. Siempre dos cubos como mínimo; con menos de 48 keyframes la
   reconstrucción falla y lo dice en pantalla.
   Entrega: los cubos, y para cada cubo qué visitas contiene.
7. **DA3 por cubo**, a resolución 504, sobre los mismos cubos.
   Entrega: la profundidad DA3 de cada keyframe.
8. **Omega por cubo**, una pasada por cubo.
   Entrega: las poses y la profundidad de Omega. Las visitas de un mismo lugar quedan reconstruidas juntas.
9. **Atada.** Costuras entre los cubos que comparten keyframes, y grafo. Los 15 m se miden con el tamaño de cada
   cubo, que sale de la profundidad DA3 de ese cubo. No hay puentes: el cierre lo dan los cubos por lugar.
   Entrega: los cubos en su lugar.
10. **Escala.** Un número, de comparar la profundidad DA3 con la profundidad de Omega ya atada.
11. **F0.** La cámara inicial, de las focales de Omega.
12. **F4.** Dos mecanismos:
    - dentro de una visita: seguir puntos por los keyframes y los testigos de F4, como hoy;
    - entre visitas de un mismo lugar: emparejar puntos guiado por la geometría del cubo, que ya dice dónde cae cada
      punto de una visita en la imagen de la otra.
    Entrega: las correspondencias, incluidas las que unen dos visitas.
13. **F5.** Ajuste de cámaras con esas correspondencias. La tolerancia sale de las costuras.
14. **F6.**
    - factor por keyframe;
    - análisis por grupos: dentro de cada lugar, cuánto está corrido el grupo de keyframes de una visita respecto de
      los demás, y corrección del grupo entero;
    - voto: un punto se publica cuando lo confirma un porcentaje de las vistas que podrían verlo, con un mínimo
      absoluto chico;
    - limpiador y publicación.

## Decisiones tomadas (con sus palabras)

- Dos pasadas, AMB3R y Omega (2026-11-22): *"prefiero mantenerlo como lo tenemos ahora, con pasada amber y pasada
  omega … no quiero comparar amber contra omega"*.
- Frames buenos (2026-11-22): *"primero limpiar los frames con blur etc, es decir dejar solo los buenos de calidad, es
  el primer paso necesario e ineludible, hay que incorporarlo"*; y *"el tema que hablamos de la obscuridad"*.
- Más keyframes por análisis espacial (2026-11-22): *"no me cierra rellenar asi porque si con separacion minima de 8
  … es temporal y nosotros tenemos analisis espacial"*; *"darle mayor cobertura … tanto al desplazamiento como al
  angulo de rotacion"*. (Reemplaza lo del 2026-11-21 sobre el hueco 425–450 de bufferStop, *"lo dejamos asi como
  esta"*: con la regla de desplazamiento ese tramo recibe keyframes.)
- Cubos por posición (2026-11-22): *"podriamos armar el cubo incluyendo kf que estan en la zona no necesariamente de
  consistencia temporal, pero si de consistencia posicional … de esa manera inevitablemente queda cerrado"*.
- El voto en porcentaje (2026-11-22): *"no es igual pedirle en f6 3 vistas a uno que tiene 5 que 4 vistas a uno que
  tiene 100, debe ser porcentual"*.
- El análisis de F6 (2026-11-22): *"necesitamos saber cuando una superficie queda mal posicionada, por que queda mal,
  y corregir el kf al ser una superficie evidentemente no es solo 1, hay varios"*.
- F4 (2026-11-22): *"hay que solucionar el tema de f4 … es importante"*.
- Sonda DA3 (2026-11-21): *"corremos el da3 sobre los 16 kf como hasta ahora al principio"*; *"puede tomar 16 kf
  distribuidos de los que haya selecionado amb3r"*. AMB3R sigue con Omega como modelo.
- PointDiT (2026-11-21): *"de momento queda afuera, no eliminado, pero si apagado"*.
- Sesión corta (2026-11-21): *"siempre 2 cubos como minimo y listo, y con minimo de 48 KF, si detecta menos de eso,
  que falle la reconstruccion, lo indica en pantalla"*.
- F2 no vuelve: nada de lo medido apunta a que falte.

## Perillas

Cada perilla de la reconstrucción, con su valor para la prueba y de dónde sale. Se completa a medida que se cierra cada
punto (USER 2026-11-22: *"vamos anotando todas las perillas que vamos teniendo para entender como se configura por
completo luego la reconstruccion"*).

### Paso 1 — frames buenos (CERRADO 2026-11-22, su "ok")

| Perilla | Antes | Valor | De dónde sale |
|---|---|---|---|
| Oscuridad: brillo medio mínimo | 40 | 0 (apagada) | En bufferStop, pccr 08-31, zaragoza y observatorio no sacaba nada. En pccr 08-24 sacaba los 144 frames de los dos tramos oscuros y abría el hueco de 130 frames del giro (cubos adivinados, el piso 36 cm abajo). Sin ella, la copia del 2026-11-19 reconstruyó el giro bien (0,93 m donde AMB3R mide 0,87). |
| Saturación: parte de píxeles quemados | 25 %, contando los ≤ 5 y los ≥ 250 | 25 %, contando solo los ≥ 250 | Los 62 frames que sacaba eran todos oscuros: contaba los casi negros, que son la oscuridad. Contando solo los quemados no saca ninguno en las cinco sesiones. |
| Quemados: brillo medio máximo | 220 | 220 | Ningún frame de las cinco sesiones pasa de 157. No saca nada en ellas; queda como guarda para un video a contraluz. |
| Nitidez: detalle frente a la mediana de sus vecinos | no existía | menos de 0,5 → se descarta | El detalle es la medida de I0 (energía de alta frecuencia). En cuatro sesiones no saca nada. En pccr 08-24 saca 12 frames: 2484–2490 y 3589–3593, los dos cruces de puerta con una superficie lisa frente a la lente (brillo 121–132, no son oscuros) — los eslabones más débiles del recorrido en el estudio del 2026-11-19. |
| Nitidez: vecinos de cada lado | — | 15 frames | Elegido para la prueba. |

Efecto: en bufferStop, pccr 08-31, zaragoza y observatorio no cambia ningún frame; en pccr 08-24 entran los frames
oscuros y salen los 12 de las puertas.

### Ya decididas en los pasos

| Paso | Perilla | Valor | De dónde sale |
|---|---|---|---|
| 3 | Sonda DA3: keyframes de AMB3R | 16, repartidos a lo largo del video | Su decisión, 2026-11-21. |
| 4, 9 | Alcance de cada vista | 15 m | Su regla (`reconstruction.view_range_m`). |
| 5 | Solape con el último keyframe | franja de 92 a 96 % | Su decisión, 2026-11-21. |
| 6 | Cubo | 64 keyframes compartiendo 32 | Su regla. |
| 6 | Sesión mínima | 48 keyframes; menos falla y lo dice en pantalla | Su decisión, 2026-11-21. |
| 7 | DA3 por cubo | resolución 504 | Su decisión, 2026-11-21. |
| 14 | PointDiT | apagado, no eliminado | Su decisión, 2026-11-21. |

## Por definir

1. ~~Paso 1 — la regla de frame malo~~ → cerrado, ver "Perillas".
2. **Paso 5 — los dos ángulos.** El de desplazamiento y el de rotación. (I1 usaba 1,77° y 10,59°, medidos en las
   imágenes; acá salen de las poses y la profundidad de AMB3R.)
3. **Paso 6 — el orden "por lugar".** En un pasillo es directo; en una sala con cruces hace falta una regla general.
   Y el límite de distancia y dirección para poner dos visitas en el mismo cubo.
4. **Paso 12 — el emparejado entre visitas.** Cómo se busca alrededor del lugar que predice la geometría, y qué pares
   de keyframes se intentan.
5. **Paso 14 — el análisis por grupos.** Qué se corrige del grupo (un factor de profundidad, una pose), contra quién
   (quién ve mejor: más cerca, mejor ángulo), y cuándo no se corrige.
6. **Paso 14 — el voto.** El porcentaje, el mínimo absoluto, y qué vistas cuentan como "podrían verlo".
7. **PointDiT.** Probarlo prendido aparte (solo F6) para ver bordes y puntos volando.

## Lo que respalda este diseño

Medido el 2026-11-22 sobre `pccr2408_prueba` y la corrida original, con el mismo método (sin placa):

- En los primeros 2000 frames del video la prueba tiene entre 25 y 38 % menos keyframes que la original, y deja hasta
  80 frames sin keyframe. De ahí en adelante tiene igual o más. Es la zona donde él ve los defectos.
- En los tres lugares que marcó hay entre 23 y 41 % menos puntos que en la original.
- En esos lugares la ida y la vuelta publican la misma superficie separadas entre 5 y 7 cm; dentro de una misma
  pasada todo coincide a 4 o 5 mm. Esa separación da igual en la original (medida por punto más cercano; falta la
  medida estricta, mismo píxel y profundidad contra profundidad).
- El factor por keyframe de F6 va de 0,69 a 1,41 en la prueba y de 0,10 a 2,31 en la original.

Del registro del proyecto:

- 2026-11-06: los cubos de espacio, por lo que ve cada keyframe, con tope de 64: costuras de 0,1 cm después del grafo
  en pccr 3108. Lo que falló el 2026-11-07 fueron los cubos sin tope de vistas (94 a 358 keyframes).
- 2026-11-08 / 09: mover el píxel que pierde el voto hasta la posición ganadora se programó y se sacó: el píxel
  conservaba su color y aparecía pintado sobre otro objeto.
- 2026-11-20 (auditoría): solo el 1,9 % de las correspondencias de F4 unía dos visitas; tres de cada cuatro ventanas
  de revisita no daban ninguna.
- 2026-11-19: los puentes entre vistas a menos de 3 m coincidían con el grafo el 77 %; a más de 6 m, el 16 % o menos.
- 2026-11-04: la nube propia de AMB3R como reconstrucción final no le ganó al pipeline.
