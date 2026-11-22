# Checklist de implementación — la arquitectura de `ARQUITECTURA_mapa.md`

USER 2026-11-22: *"ok, y armamos checklist para la implementacion"*.

Cómo se trabaja (sus reglas):
- Una línea por vez. Antes de empezar una línea se la nombra y se espera su OK. Nada se lanza sin su OK.
- Cada línea dice qué tiene que mostrarse para cerrarla. Se cierra con una comprobación escrita, cuya salida se le
  muestra; nunca "terminado" con algo pendiente.
- El código queda en `experiments/new_sequence/`, fuera de la línea principal, hasta que él lo valide en las sesiones
  (como el candidato del 2026-11-21).

---

- [x] **1. Paso 1 — frames buenos.** Las cuatro reglas (quemados, saturados contando solo los ≥ 250, nitidez frente a
      15 + 15 vecinos; sin oscuridad).
      *Cierre:* en las cinco sesiones, los frames descartados son los 12 de las puertas de pccr 08-24 y ningún otro.
      ✓ 2026-11-22 — `scripts/checklist/check_arq_1_frames.sh` → CLOSED. `arq3_frames.py` en placa (el Y del JPEG por
      nvJPEG, brillo y quemados como cuentas enteras, la reducción INTER_AREA de OpenCV en su mismo orden, la FFT en
      float64) contra las funciones de I0, en los 20.279 frames de las cinco sesiones: las mismas decisiones, brillo y
      quemados idénticos, las imágenes reducidas idénticas bit a bit, la nitidez a 1e-15. Descartados: 2484–2490 y
      3589–3593 de pccr 08-24, por detalle; ninguno más. 28 s de placa para los 20.279 frames. Encontrado en la primera
      corrida y corregido: dividido por un número de Python, PyTorch en placa multiplica por el inverso (un bit de
      diferencia a veces); se divide por un tensor en la placa.
- [ ] **2. Paso 3 — sonda DA3 sobre 16 keyframes propios de AMB3R.**
      *Cierre:* el metro de AMB3R y la cámara de las tres sesiones de prueba, junto al metro y la cámara que tienen hoy.
- [ ] **3. Paso 4 — el mapa.** Sectores, visibilidad por rayos con la tolerancia de la sesión.
      *Cierre:* el mapa de las tres sesiones, con su tamaño de sector y su tolerancia medidos, y en planta, para su
      ojo, qué keyframes ven una muestra de lugares (incluido uno tapado por una columna).
- [ ] **4. Paso 5 — selección de keyframes** con las tres medidas sobre el mapa.
      *Cierre:* los keyframes de las tres sesiones, por tramo del video, contra la original y la prueba del 21/11 (el
      principio de pccr 08-24 en particular); ningún frame malo elegido.
- [ ] **5. Paso 6 — cubos por continuidad**, el mínimo de 48 keyframes y su mensaje en pantalla.
      *Cierre:* los cubos de las tres sesiones (64 compartiendo 32), y una sesión de menos de 48 keyframes que falla
      con el mensaje.
- [ ] **6. Pasos 7 y 8 — DA3 y Omega por cubo** sobre esos cubos.
      *Cierre:* corren en bufferStop y dejan, por keyframe y cubo, la versión de Omega y la profundidad DA3.
- [ ] **7. Paso 9 — MASt3R en el pod.** Su código y sus pesos instalados (un cambio del pod: con su OK, y después el
      kit de reconstrucción al día).
      *Cierre:* MASt3R corre en esta placa sobre un par de bufferStop; el kit de reconstrucción dice que es el pod.
- [ ] **8. Paso 9 — pares y correspondencias.** Los pares del mapa y MASt3R sobre cada uno.
      *Cierre:* en bufferStop, cuántos pares, cuánto tarda cada par y cuántas correspondencias da cada uno.
- [ ] **9. Paso 9 — verificación, ajuste fino y trazas.**
      *Cierre:* las trazas de bufferStop, con cuántas correspondencias pasan la verificación contra el mapa y cuántas
      unen dos visitas.
- [ ] **10. Paso 10 — ajuste, primera etapa (3D).**
      *Cierre:* en bufferStop, cuánto coinciden las versiones de los keyframes compartidos después de la alineación,
      junto a las costuras de hoy (1,6–2,7 cm en la prueba del 21/11).
- [ ] **11. Paso 10 — ajuste, segunda etapa (reproyección)**, con las anclas cada 8 píxeles, la cámara única y los
      testigos; sale una pose y un mapa por keyframe.
      *Cierre:* en bufferStop, el error de reproyección con datos reservados, junto al de F5 de hoy (0,2998 px en la
      prueba del 21/11).
- [ ] **12. Paso 11 — escala.**
      *Cierre:* el recorrido de bufferStop en metros, junto a los 12,94 m de la corrida validada.
- [ ] **13. Paso 12 — F6:** PointDiT prendido, el voto del 60 % con mínimo 3 sobre las vistas del mapa, limpiador y
      publicación.
      *Cierre:* la nube de bufferStop publicada en una sesión de prueba.
- [ ] **14. Corridas completas.** bufferStop, pccr 08-31 y pccr 08-24, cada una con su OK.
      *Cierre:* las tres nubes publicadas y su ojo sobre cada una.
