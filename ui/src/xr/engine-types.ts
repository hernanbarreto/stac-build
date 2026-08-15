// STAC-Builder — shared types for the mobile XR engines.
// Hernán Barreto - Ingerop IN3 Session IV - STAC

export type Tool = 'move' | 'dist' | 'angle' | 'vol'

// METRIC FIRST (user mandate): 1:1 is the default; miniatures are the option
export const SCALES: Array<[number, string]> = [[1, '1:1'], [0.1, '1:10'], [0.02, '1:50']]

export interface EngineCallbacks {
  onReady: () => void
  onError: (msg: string) => void
  onToast: (msg: string) => void
  onPlaced: () => void
  onTracking: (status: string) => void
}

export interface IXREngine {
  placed: boolean
  tool: Tool
  scaleIdx: number
  start(canvas: HTMLCanvasElement): void
  stop(): void
  setTool(tool: Tool): void
  setScaleIdx(i: number): string
  recenter(): void
  clearMeasures(): void
  /** screen tap in non-immersive contexts; returns a short status string */
  tap(clientX: number, clientY: number): string | null
}

export function tele(event: string, data: Record<string, unknown> = {}) {
  try {
    fetch('/api/ar/log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event: `xr-${event}`, ...data }),
    }).catch(() => {})
  } catch { /* telemetry must never break the app */ }
}

// XR browsers have no devtools: EVERY uncaught exception must reach the pod
// log, or failures die silently
window.addEventListener('error', (e) =>
  tele('js-error', { msg: String(e.message), src: `${e.filename}:${e.lineno}` }))
window.addEventListener('unhandledrejection', (e) =>
  tele('promise-rejection', { msg: String((e as any).reason).slice(0, 300) }))
