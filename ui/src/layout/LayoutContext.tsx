/**
 * LayoutContext — the shell's geometry and preferences (prompt_ui.txt §5).
 *
 * Holds: which left-panel tab is open (or none), its width; whether the
 * inspector is open, its tab and width; whether the bottom dock is open and
 * its tab; the density. Everything is persisted in localStorage under one
 * key (Electron/desktop app — §15 allows it for layout, density, language).
 *
 * The previous keys `stac.assistantOpen` / `stac.assistantWidth` are read
 * once as a migration so a user keeps their dock width.
 *
 * Density default: compact on a fine pointer, comfortable when the primary
 * pointer is coarse (tablet). Applied to <html data-density>.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

export type LeftTab = 'sessions' | 'instances' | 'tickets' | 'scans' | 'bim' | 'docs' | 'gantt' | 'globe' | 'team'
export type InspectorTab = 'properties' | 'assistant'
export type DockTab = 'console' | 'jobs' | 'timeline' | 'evaluation'
export type Density = 'compact' | 'comfortable'

export interface LayoutState {
  leftTab: LeftTab | null
  leftWidth: number
  inspectorOpen: boolean
  inspectorTab: InspectorTab
  inspectorWidth: number
  dockOpen: boolean
  dockTab: DockTab
  dockHeight: number
  density: Density
}

const STORAGE_KEY = 'stac.layout.v1'
const LIMITS = { leftMin: 200, leftMax: 640, inspMin: 260, inspMax: 720, dockMin: 120, dockMax: 600 }

function defaultDensity(): Density {
  try {
    return window.matchMedia('(pointer: coarse)').matches ? 'comfortable' : 'compact'
  } catch { return 'compact' }
}

function defaults(): LayoutState {
  let inspectorOpen = true
  let inspectorWidth = 320
  try {
    const legacyOpen = localStorage.getItem('stac.assistantOpen')
    if (legacyOpen != null) inspectorOpen = legacyOpen !== '0'
    const legacyW = Number(localStorage.getItem('stac.assistantWidth'))
    if (legacyW) inspectorWidth = legacyW
  } catch { /* storage unavailable */ }
  // (a first visit on a page where both panels cannot be open: the sessions' list, not the assistant, is shown)
  if (oneAtATime({ leftWidth: 280, inspectorWidth })) inspectorOpen = false
  return {
    leftTab: 'sessions',
    leftWidth: 280,
    inspectorOpen,
    inspectorTab: 'assistant',
    inspectorWidth,
    dockOpen: false,
    dockTab: 'console',
    dockHeight: 200,
    density: defaultDensity(),
  }
}

function load(): LayoutState {
  const d = defaults()
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return d
    const saved = JSON.parse(raw) as Partial<LayoutState>
    // (the report was a tab of the dock until 2026-11-19: it lives in the BIM panel)
    if ((saved.dockTab as string | undefined) === 'report') saved.dockTab = d.dockTab
    // (the certificate panel — the inspector's 'acta' tab — left the interface on 2026-11-19)
    if ((saved.inspectorTab as string | undefined) === 'acta') saved.inspectorTab = 'properties'
    const s = { ...d, ...saved }
    // (a layout stored on a wide screen, opened on a narrow one: the left panel — where a session is chosen — wins)
    if (s.leftTab && s.inspectorOpen && oneAtATime(s)) s.inspectorOpen = false
    return s
  } catch { return d }
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

// ── THE PAGE'S SIZE (USER 2026-11-20: "hace la pagina responsiva para PC tablet y telefonos, con cortes por
// resolucion de pantalla"; the cuts he approved). Under OVERLAY_MAX_W the side panels open OVER the viewer
// (styles/layout.css); the two are then kept open together only while a strip of the viewer of at least
// MIN_VIEWER_STRIP stays between them — a tablet on its side keeps both, a tablet upright one at a time.
const OVERLAY_MAX_W = 1279
const MIN_VIEWER_STRIP = 240
const RAIL_W = 48
const pageWidth = () => { try { return window.innerWidth } catch { return 1920 } }
/** the side panels open over the viewer at this page width */
export const panelsOverlay = (w = pageWidth()) => w <= OVERLAY_MAX_W
/** both panels cannot be open together at this page width (with these panel widths) */
const oneAtATime = (s: Pick<LayoutState, 'leftWidth' | 'inspectorWidth'>, w = pageWidth()) =>
  panelsOverlay(w) && w - RAIL_W - s.leftWidth - s.inspectorWidth < MIN_VIEWER_STRIP

interface LayoutContextValue {
  state: LayoutState
  setLeftTab: (tab: LeftTab | null) => void
  toggleLeftTab: (tab: LeftTab) => void
  setLeftWidth: (w: number) => void
  setInspectorOpen: (open: boolean) => void
  toggleInspector: () => void
  setInspectorTab: (tab: InspectorTab) => void
  openInspector: (tab: InspectorTab) => void
  setInspectorWidth: (w: number) => void
  setDockOpen: (open: boolean) => void
  toggleDock: () => void
  setDockTab: (tab: DockTab) => void
  openDock: (tab: DockTab) => void
  setDockHeight: (h: number) => void
  setDensity: (d: Density) => void
  /** the page is too narrow for both side panels: opening one closes the other (asked when it matters, so that a
   *  resized window is never answered from an old measure) */
  panelsExclusive: () => boolean
}

const LayoutContext = createContext<LayoutContextValue | null>(null)

export function LayoutProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<LayoutState>(load)

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)) } catch { /* storage unavailable */ }
  }, [state])

  useEffect(() => {
    document.documentElement.dataset.density = state.density
  }, [state.density])

  const patch = useCallback((p: Partial<LayoutState> | ((s: LayoutState) => Partial<LayoutState>)) => {
    setState(s => ({ ...s, ...(typeof p === 'function' ? p(s) : p) }))
  }, [])

  // THE ON-SCREEN KEYBOARD (a touch screen): it covers the bottom of the page without making the page shorter
  // (an iPhone never does; the assistant's question field sat under it). The part in sight is given to the
  // frame as --page-h (styles/layout.css). Only while the page is not zoomed — a pinch is not a keyboard.
  useEffect(() => {
    const vv = window.visualViewport
    let coarse = false
    try { coarse = window.matchMedia('(pointer: coarse)').matches } catch { /* no media queries */ }
    if (!vv || !coarse) return
    const root = document.documentElement
    const set = () => {
      if (Math.abs(vv.scale - 1) < 0.01) root.style.setProperty('--page-h', `${Math.round(vv.height)}px`)
      else root.style.removeProperty('--page-h')
    }
    set()
    vv.addEventListener('resize', set)
    return () => { vv.removeEventListener('resize', set); root.style.removeProperty('--page-h') }
  }, [])

  // a page that became too narrow for both panels (a rotated tablet, a shrunk window): the one opened last is
  // not known here — the right one gives way, as at load
  useEffect(() => {
    const onResize = () => setState(s => (s.leftTab && s.inspectorOpen && oneAtATime(s) ? { ...s, inspectorOpen: false } : s))
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  const value = useMemo<LayoutContextValue>(() => ({
    state,
    // (opening one panel closes the other where both do not fit — oneAtATime)
    setLeftTab: tab => patch(s => ({ leftTab: tab, ...(tab && oneAtATime(s) ? { inspectorOpen: false } : {}) })),
    toggleLeftTab: tab => patch(s => (s.leftTab === tab ? { leftTab: null } : { leftTab: tab, ...(oneAtATime(s) ? { inspectorOpen: false } : {}) })),
    setLeftWidth: w => patch({ leftWidth: clamp(w, LIMITS.leftMin, LIMITS.leftMax) }),
    setInspectorOpen: open => patch(s => ({ inspectorOpen: open, ...(open && oneAtATime(s) ? { leftTab: null } : {}) })),
    toggleInspector: () => patch(s => (s.inspectorOpen ? { inspectorOpen: false } : { inspectorOpen: true, ...(oneAtATime(s) ? { leftTab: null } : {}) })),
    setInspectorTab: tab => patch({ inspectorTab: tab }),
    openInspector: tab => patch(s => ({ inspectorOpen: true, inspectorTab: tab, ...(oneAtATime(s) ? { leftTab: null } : {}) })),
    setInspectorWidth: w => patch({ inspectorWidth: clamp(w, LIMITS.inspMin, LIMITS.inspMax) }),
    setDockOpen: open => patch({ dockOpen: open }),
    toggleDock: () => patch(s => ({ dockOpen: !s.dockOpen })),
    setDockTab: tab => patch({ dockTab: tab }),
    openDock: tab => patch({ dockOpen: true, dockTab: tab }),
    setDockHeight: h => patch({ dockHeight: clamp(h, LIMITS.dockMin, LIMITS.dockMax) }),
    setDensity: d => patch({ density: d }),
    panelsExclusive: () => oneAtATime(state),
  }), [state, patch])

  return <LayoutContext.Provider value={value}>{children}</LayoutContext.Provider>
}

export function useLayout(): LayoutContextValue {
  const ctx = useContext(LayoutContext)
  if (!ctx) throw new Error('useLayout must be used inside <LayoutProvider>')
  return ctx
}

/** Shared drag-to-resize helper for the region handles — by mouse, pen or FINGER (pointer events: a tablet could
 *  not resize a panel; USER 2026-11-20, the page on every screen). The strip carries `touch-action: none`. */
export function startDrag(
  e: React.PointerEvent,
  axis: 'x' | 'y',
  start: number,
  sign: 1 | -1,
  onChange: (v: number) => void,
  onDone?: () => void,
) {
  if (e.pointerType === 'mouse' && e.button !== 0) return
  e.preventDefault()
  const id = e.pointerId
  const origin = axis === 'x' ? e.clientX : e.clientY
  const onMove = (ev: PointerEvent) => {
    if (ev.pointerId !== id) return
    const cur = axis === 'x' ? ev.clientX : ev.clientY
    onChange(start + sign * (cur - origin))
  }
  const onUp = (ev: PointerEvent) => {
    if (ev.pointerId !== id) return
    document.removeEventListener('pointermove', onMove)
    document.removeEventListener('pointerup', onUp)
    document.removeEventListener('pointercancel', onUp)
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
    onDone?.()
  }
  document.addEventListener('pointermove', onMove)
  document.addEventListener('pointerup', onUp)
  document.addEventListener('pointercancel', onUp)
  document.body.style.cursor = axis === 'x' ? 'col-resize' : 'row-resize'
  document.body.style.userSelect = 'none'
}
