# Configuración de Alto Rendimiento: Firefox para STAC-Build

Esta guía detalla los pasos necesarios para configurar Mozilla Firefox como el navegador principal de desarrollo y ejecución para STAC-Build. 

El objetivo es evitar los cuellos de botella de memoria (típicos de navegadores basados en Chromium) y forzar al sistema operativo a asignar el 100% de los recursos de la GPU dedicada para renderizar nubes de puntos masivas (Potree/WebGL) y superposiciones 2D.

---

## FASE 1: Configuración a nivel Sistema Operativo (Windows)
*Objetivo: Obligar a Windows a usar la GPU dedicada (ej. NVIDIA RTX) en lugar de la gráfica integrada del procesador.*

1. **Ubicar el ejecutable de Firefox:**
   * Ruta principal: `C:\Program Files\Mozilla Firefox\firefox.exe`
   * *(Opcional)* Si no está ahí, clic derecho en el acceso directo de Firefox > **Abrir ubicación del archivo**. Copiar la ruta.

2. **Forzar el Alto Rendimiento en Windows:**
   * Presionar la tecla `Windows` y buscar **Configuración de gráficos** (o navegar a *Sistema > Pantalla > Gráficos*).
   * En "Agregar una aplicación", seleccionar **Aplicación de escritorio** y hacer clic en **Examinar**.
   * Pegar la ruta copiada y seleccionar `firefox.exe`.
   * Una vez que Firefox aparezca en la lista, hacer clic en **Opciones**.
   * Seleccionar **Alto rendimiento** (debe indicar el nombre de la GPU dedicada).
   * Hacer clic en **Guardar**.

---

## FASE 2: Configuración del Motor Interno (Firefox `about:config`)
*Objetivo: Desbloquear los límites de seguridad de memoria y WebGL para manejar los arreglos matemáticos pesados de la fotogrametría.*

1. Abrir Firefox.
2. En la barra de direcciones, escribir `about:config` y presionar Enter.
3. Aceptar la advertencia de seguridad ("Aceptar el riesgo y continuar").
4. Usar la barra de búsqueda superior para modificar las siguientes variables:

### 1. Forzar renderizado por hardware extremo
* **Variable:** `gfx.webrender.all`
* **Acción:** Cambiar a `true`
* **Por qué:** Obliga a Firefox a dibujar absolutamente todo usando la arquitectura de la GPU dedicada.

### 2. Desatar WebGL sin restricciones
* **Variable:** `webgl.force-enabled`
* **Acción:** Cambiar a `true`
* **Por qué:** Evita que el navegador bloquee funciones avanzadas de WebGL al detectar cargas inusuales de millones de puntos.

### 3. Aislar el proceso 3D (Escudo Anti-Crashes)
* **Variable:** `webgl.out-of-process`
* **Acción:** Cambiar a `true`
* **Por qué:** Ejecuta el cálculo espacial en un núcleo separado. Si Potree satura la memoria, la pestaña puede reiniciarse sin congelar todo el navegador.

### 4. Aumentar el tiempo de ejecución de scripts pesados
* **Variable:** `dom.max_script_run_time`
* **Acción:** Editar el valor numérico y cambiarlo a `60`
* **Por qué:** Evita que Firefox interrumpa el cálculo del Octree espacial o el cruce de datos BIM mostrando el error de "Página ralentizando el navegador".

---

## FASE 3: Aplicar Cambios
* Cerrar Firefox por completo y volver a iniciarlo para que el motor WebGL y el sistema operativo tomen las nuevas directivas.
* Cargar el entorno local de STAC-Build y verificar los FPS durante el paneo de la nube de puntos.