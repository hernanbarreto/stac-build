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

export type LeftTab = 'sessions' | 'instances' | 'scans' | 'bim' | 'team'
export type InspectorTab = 'properties' | 'acta' | 'assistant'
export type DockTab = 'console' | 'jobs' | 'timeline' | 'report'
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
    return { ...d, ...saved }
  } catch { return d }
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

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

  const value = useMemo<LayoutContextValue>(() => ({
    state,
    setLeftTab: tab => patch({ leftTab: tab }),
    toggleLeftTab: tab => patch(s => ({ leftTab: s.leftTab === tab ? null : tab })),
    setLeftWidth: w => patch({ leftWidth: clamp(w, LIMITS.leftMin, LIMITS.leftMax) }),
    setInspectorOpen: open => patch({ inspectorOpen: open }),
    toggleInspector: () => patch(s => ({ inspectorOpen: !s.inspectorOpen })),
    setInspectorTab: tab => patch({ inspectorTab: tab }),
    openInspector: tab => patch({ inspectorOpen: true, inspectorTab: tab }),
    setInspectorWidth: w => patch({ inspectorWidth: clamp(w, LIMITS.inspMin, LIMITS.inspMax) }),
    setDockOpen: open => patch({ dockOpen: open }),
    toggleDock: () => patch(s => ({ dockOpen: !s.dockOpen })),
    setDockTab: tab => patch({ dockTab: tab }),
    openDock: tab => patch({ dockOpen: true, dockTab: tab }),
    setDockHeight: h => patch({ dockHeight: clamp(h, LIMITS.dockMin, LIMITS.dockMax) }),
    setDensity: d => patch({ density: d }),
  }), [state, patch])

  return <LayoutContext.Provider value={value}>{children}</LayoutContext.Provider>
}

export function useLayout(): LayoutContextValue {
  const ctx = useContext(LayoutContext)
  if (!ctx) throw new Error('useLayout must be used inside <LayoutProvider>')
  return ctx
}

/** Shared drag-to-resize helper for the region handles. */
export function startDrag(
  e: React.MouseEvent,
  axis: 'x' | 'y',
  start: number,
  sign: 1 | -1,
  onChange: (v: number) => void,
  onDone?: () => void,
) {
  e.preventDefault()
  const origin = axis === 'x' ? e.clientX : e.clientY
  const onMove = (ev: MouseEvent) => {
    const cur = axis === 'x' ? ev.clientX : ev.clientY
    onChange(start + sign * (cur - origin))
  }
  const onUp = () => {
    document.removeEventListener('mousemove', onMove)
    document.removeEventListener('mouseup', onUp)
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
    onDone?.()
  }
  document.addEventListener('mousemove', onMove)
  document.addEventListener('mouseup', onUp)
  document.body.style.cursor = axis === 'x' ? 'col-resize' : 'row-resize'
  document.body.style.userSelect = 'none'
}
