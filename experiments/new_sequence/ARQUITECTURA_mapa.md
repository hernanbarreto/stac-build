# Arquitectura de reconstrucción guiada por el mapa de AMB3R — CERRADA (2026-11-22)

Documento de diseño, trabajado con él punto por punto el 2026-11-22 y cerrado con su OK (*"ok, y armamos checklist
para la implementacion"*). No está programado. La implementación sigue `CHECKLIST_implementacion.md`. Las versiones
anteriores de este documento están en la historia de git.

Antecedente: `CHECKLIST_arquitectura.md` (la prueba del 2026-11-21 en tres sesiones, sus números y sus veredictos).

---

## El principio

- *"prefiero mantenerlo como lo tenemos ahora, con pasada amber y pasada omega, pero de forma mucho mas inteligente
  … sacarle todo el jugo a lo que ya tenemos de la pasada amber, para todas las etapas … todo debe estar respaldado
  por la nube y poses y campos de vision iniciales"*.
- *"no estamos ciegos, no hay que inventar nada, ya lo tenemos"*.
- De cada keyframe sale de la etapa de corrección **una sola pose y un solo mapa de profundidad**: *"cuando salimos a
  los pasos F0 a F6 cada kf debe tener una unica pose un unico mapa"*.

---

## Fase I · Estructura global

### 1. Frames buenos
Un frame se descarta cuando su brillo medio pasa de 220 (quemado), cuando más del 25 % de sus píxeles están quemados
(≥ 250) (saturado), o cuando su detalle es menos de la mitad de la mediana del de sus vecinos (15 frames de cada
lado). La oscuridad no descarta: un tramo oscuro se queda.
- Entra: todos los frames del video.
- Sale: la lista de frames buenos. Solo un frame bueno puede ser keyframe o testigo. AMB3R lee igual todos los
  frames: sigue la cámara frame a frame.

### 2. AMB3R sobre todos los frames
AMB3R-SLAM sobre el video completo, con Omega como modelo.
- Entra: todos los frames.
- Sale: la pose de todos los frames, la profundidad de uno de cada cuatro y sus keyframes propios.

### 3. Sonda DA3
DA3 sobre 16 keyframes propios de AMB3R, repartidos a lo largo del video.
- Sale: el metro de AMB3R (para los 15 m del mapa y de la selección) y la cámara de la sesión (su campo de visión
  fija el umbral de rotación del paso 5). No es el metro de la nube: ese es el paso 11.

---

## Fase II · Mapa y organización

### 4. El mapa
La nube de AMB3R en metros, hasta 15 m, dividida en sectores. Cada frame, desde su propia pose, ve un sector solo si
es la primera superficie que encuentran sus rayos, dentro de la tolerancia de la sesión: lo tapado no cuenta, lo
visible sí, aunque sea en parte.
- Sale: para cada sector, qué frames lo ven, desde qué distancia y desde qué dirección. Todo lo que sigue lo lee: la
  selección, los pares de correspondencias y las vistas que cuentan en el voto.

### 5. Selección de keyframes
Se recorre el video desde el primer frame bueno. Para cada frame se miden tres cosas respecto del último keyframe:
- **solape** de la vista: banda de 92 a 96 %, cierra cuando baja de 92 %;
- **desplazamiento**: lo que queda del movimiento de cada punto después de descontar el giro que mejor explica la
  vista (la medida de I1, sobre los puntos de AMB3R), cuantil 0,9; umbral 1,26°, banda ±25 %;
- **rotación**: el giro entre las orientaciones de las dos cámaras; umbral 0,7 × campo horizontal / 4, banda ±25 %.

Un frame es candidato cuando cualquiera de las tres está dentro de su banda; la ventana se cierra cuando cualquiera se
pasa de su banda; el keyframe es el candidato bueno más nítido. Si una medida salta toda su banda entre dos frames
seguidos, se toman los dos. No hay tope temporal ni relleno.
- Sale: los keyframes, y los testigos: los frames buenos que quedan entre keyframes.

### 6. Cubos por continuidad
Los keyframes en orden de tiempo, de a 64, compartiendo 32 con el cubo siguiente. Siempre dos cubos como mínimo: con
menos de 48 keyframes la reconstrucción falla y lo dice en pantalla (información insuficiente, no se puede reconstruir
con este método).
- Sale: los cubos.
- Los cubos por posición quedan como alternativa (ver "Alternativa"), si el ajuste del paso 10 no cierra bien las
  visitas.

---

## Fase III · Observaciones

### 7. DA3 por cubo
DA3 sobre los keyframes de cada cubo juntos, a resolución 504.
- Sale: la profundidad DA3 de cada keyframe. Sirve para el tamaño de cada cubo (los 15 m en el paso 10) y para el
  metro (paso 11). DA3 no entra como restricción del ajuste: *"da3 metro nada mas"*.

### 8. Omega por cubo
Una pasada de Omega por cubo, con sus keyframes.
- Sale: para cada keyframe y cada cubo donde está, una versión de su pose y de su profundidad, con la confianza de ese
  cubo.

### 9. Correspondencias
1. **Pares**, del mapa: cada keyframe con todos los que ven lo mismo, sean del mismo cubo, de cubos vecinos o de otra
   visita; y cada testigo con sus keyframes vecinos. No se descarta ninguno.
2. **MASt3R** sobre cada par: qué píxel de una imagen corresponde a qué píxel de la otra, con su confianza.
3. **Verificación contra el mapa:** una correspondencia tiene que caer donde el mapa la predice, dentro de la
   tolerancia de la sesión.
4. **Ajuste fino:** la etapa fina del rastreador de F4 lleva cada correspondencia a la resolución nativa.
5. **Trazas:** las correspondencias se juntan en trazas de varias vistas.
- Sale: las trazas, cada observación con su confianza y su procedencia.

---

## Fase IV · Ajuste conjunto

### 10. Ajuste conjunto
Reemplaza a la atada, a F5 y al factor por keyframe de F6.
- **Entra:** las versiones de Omega de cada keyframe (paso 8), cada una pesada por la confianza de su cubo; las
  trazas (paso 9); las poses de AMB3R, como punto de partida; el tamaño de cada cubo por DA3 (los 15 m).
- **Incógnitas:** la posición y la escala de cada cubo; la pose de cada keyframe; la pose de cada testigo que aporta
  trazas; una cámara para toda la sesión (focal y distorsión); y la corrección de profundidad de cada keyframe: anclas
  cada 8 píxeles, en profundidad inversa o logarítmica, sobre su forma base.
- **Forma base de la profundidad de un keyframe:** sus versiones; las que coinciden dentro de la tolerancia de la
  sesión se combinan (es ruido); donde no coinciden, se queda la de más confianza. Nunca se promedian versiones que no
  coinciden, ni se fuerza una sobre otra.
- **Primera etapa:** alineación en 3D de los puntos que se corresponden, pesada por confianza.
- **Segunda etapa:** reproyección en la imagen, con pérdida robusta; mueve poses, cámara y anclas.
- **Sistema de referencia:** la pose del primer keyframe fija; la vertical, la del paso de orientación; la escala la
  fija el paso 11.
- **Sale:** una pose y un mapa de profundidad por keyframe; la cámara; la pose de los testigos; el residuo de cada
  observación.

### 11. Escala
Un número: la profundidad DA3 (paso 7) contra la profundidad ajustada, en la banda cercana y hasta 15 m. Se aplica a
las poses y a las profundidades. Es el metro de la nube, no una validación.

---

## Fase V · Publicación

### 12. F6
- **PointDiT prendido:** el detalle en los bordes.
- **Voto:** un punto se publica cuando lo confirma al menos el 60 % de las vistas que, según el mapa, pueden verlo, con
  un mínimo de 3 confirmaciones. Una vista que tiene algo adelante no cuenta. El veto por oclusión sigue como hoy.
- **Limpiador y publicación,** como hoy.

---

## Perillas

| Paso | Perilla | Valor | De dónde sale |
|---|---|---|---|
| 1 | Oscuridad: brillo medio mínimo | 0 (apagada) | Medido: en pccr 08-24 sacaba los 144 frames de los dos tramos oscuros y abría el hueco del giro; en las otras cuatro sesiones no sacaba nada. |
| 1 | Saturación: parte de píxeles quemados | 25 %, contando solo los ≥ 250 | Los 62 frames que sacaba eran todos oscuros (contaba los casi negros). Contando solo los quemados no saca ninguno en las cinco sesiones. |
| 1 | Quemados: brillo medio máximo | 220 | Ningún frame de las cinco sesiones pasa de 157; queda como guarda. |
| 1 | Nitidez: detalle frente a la mediana de los vecinos | menos de 0,5 → se descarta | En cuatro sesiones no saca nada; en pccr 08-24 saca los 12 frames de los dos cruces de puerta (2484–2490, 3589–3593). |
| 1 | Nitidez: vecinos de cada lado | 15 frames | Elegido para la prueba. |
| 3 | Sonda DA3 | 16 keyframes propios de AMB3R, repartidos | Su decisión, 2026-11-21. |
| 4, 5, 9, 12 | Alcance de cada vista | 15 m | Su regla. |
| 4 | Sector del mapa | tolerancia de la sesión × distancia típica (≈ 5 cm en bufferStop) | Medido en cada sesión. |
| 4 | Tolerancia de visibilidad | la de la sesión (2,4 % en bufferStop) | Lo que AMB3R coincide consigo mismo entre keyframes seguidos, medido en cada sesión. |
| 5 | Solape | banda 92–96 %; cierra bajo 92 % | Medido en la corrida validada de bufferStop: mediana 93,7 % entre keyframes seguidos. |
| 5 | Desplazamiento | 1,26°, cuantil 0,9 | Medido: la mediana de esta medida en las ventanas que I1 cerró por desplazamiento, en las tres corridas validadas (bufferStop 1,40°, pccr 08-31 1,23°, pccr 08-24 1,26°). Su OK. |
| 5 | Rotación | 0,7 × campo horizontal / 4 (10,8° en pccr) | El de I1, su elección del 2026-10-07. Medida sobre las poses de AMB3R coincide con I1 (11,3° / 11,3°; 9,3° / 11,8°; 9,1° / 9,9°). |
| 5 | Banda de desplazamiento y de rotación | ±25 % | El de I1. |
| 6 | Cubo | 64 keyframes compartiendo 32 | Su regla. |
| 6 | Sesión mínima | 48 keyframes, dos cubos | Su decisión, 2026-11-21. |
| 7 | DA3 por cubo | resolución 504 | Su decisión, 2026-11-21. |
| 9 | Pares de MASt3R | todos los que el mapa diga que ven lo mismo | Su decisión: *"todos los necesarios, no se descarta nada"*. |
| 10 | Corrección de profundidad | anclas cada 8 píxeles | Su decisión (MASt3R-SfM). |
| 12 | Voto | 60 % de las vistas que pueden ver el punto | Su decisión. Medido: publica 0,99 (bufferStop) y 0,98 (pccr 08-31) de lo que se publica hoy. |
| 12 | Voto: mínimo | 3 confirmaciones | Su decisión. |
| 12 | PointDiT | prendido | Su decisión: *"si que se incluye en el pipeline"*. |

---

## Lo que se tomó de las propuestas de GPT y de Gemini (2026-11-22)

De GPT: escribirlo por fases, con lo que entra y sale de cada paso; una sola escala para toda la nube; un solo ajuste
para todo, aunque por memoria se resuelva por bloques; cada observación guarda su procedencia, su confianza y su
residuo, porque con eso se pesa; los testigos se ajustan sin pasar a ser keyframes.

De Gemini: juntar las correspondencias en trazas de varias vistas; verificar cada correspondencia contra el mapa; las
anclas en profundidad inversa o logarítmica; una sola cámara para la sesión; fijar el sistema de referencia.

Lo que no se tomó, y por qué:
- **Menos keyframes (solape de 70–85 %):** medido al revés; con menos keyframes la nube salió peor, y el principio de
  pccr 08-24 falló justo donde había menos.
- **Descartar frames estáticos antes de AMB3R:** AMB3R necesita todos los frames; los estáticos ya los saltea la
  selección.
- **DA3 una vez por keyframe:** medido, frame por frame dio la nube 7,5 % más chica y más ruidosa.
- **DA3 como restricción del ajuste, o la corrección a·D + b:** su regla (DA3 solo da el metro); medido, las
  restricciones de DA3 por cubo rompieron costuras de 16 a 33 cm; la corrección es con anclas cada 8 píxeles.
- **Una escala solo como indicador:** un cálculo que no corrige nada (su regla); el paso 11 es el metro y se aplica.
- **TSDF:** promedia superficies.
- **Criterios sin definir** (calidad por regiones, "continuidad espacial", filtros de correspondencias sin umbral,
  "consolidación de sectores"): no tienen medida ni valor.

---

## Alternativa: cubos por posición

Si el ajuste del paso 10 no cierra bien las visitas, se agregan cubos por posición, que Omega reconstruye junto con los
de continuidad, todos en el mismo ajuste:
- un cubo por cada lugar revisitado, con los keyframes de todos los momentos que lo ven;
- dos keyframes ven el mismo lugar en otra visita cuando están a más de 32 keyframes en el recorrido, sus cámaras están
  a menos de 3 m y comparten una parte de lo que ven (con los rayos del mapa);
- el lugar arranca en celdas de 5 m y se parte a la mitad mientras lo vean más de 64 keyframes; un keyframe entra al
  cubo de un lugar cuando al menos el 30 % de lo que ve cae en él.

De dónde salen esos valores: los 3 m, de los puentes del 2026-11-19 (Omega con dos visitas coincidía con el grafo el
76–79 % hasta 3 m, y caía después); los 5 m y el tope de 64, de los cubos de espacio del 2026-11-06 (costuras de 0,1 cm
en pccr 08-31); el 30 %, de esos mismos cubos (*"no tiene sustento el 30% que deberia ser una perilla, pero sirve"*).

---

## Lo que respalda este diseño

Medido el 2026-11-22 sobre `pccr2408_prueba` y la corrida original, con el mismo método:
- En los primeros 2000 frames la prueba tenía entre 25 y 38 % menos keyframes que la original, y hasta 80 frames sin
  keyframe; ahí él vio los defectos (capas duplicadas, superficies dobladas, puntos volando).
- En los lugares que marcó, la ida y la vuelta publican la misma superficie separadas entre 5 y 7 cm; dentro de una
  pasada todo coincide a 4–5 mm.

Del registro y de la bibliografía:
- 2026-11-20 (auditoría): solo el 1,9 % de las correspondencias de F4 unía dos visitas; tres de cada cuatro ventanas
  de revisita no daban ninguna. MASt3R encuentra correspondencias entre vistas muy distintas.
- 2026-11-20: forzar a coincidir las versiones de un keyframe en cadena rompió los cubos (la escala se fue 30 % en
  cinco cubos); de ahí la regla de no forzar ni promediar.
- MASt3R-SfM (NAVER, 2024): ajuste en dos etapas (3D y reproyección), profundidad con anclas cada 8 píxeles, una pose
  y una profundidad por imagen; escenas de más de 1000 imágenes.
- AMB3R contra MASt3R-SfM (paper de AMB3R, CVPR 2026): mejores poses en ETH3D y Tanks & Temples; la profundidad en
  metros de MASt3R tiene 57–64 % de error contra 5–9 % de AMB3R: el metro de MASt3R no sirve.
