/**
 * STAC Build — Main Application
 * Hernán Barreto — Ingerop IN3
 *
 * State and business logic live here unchanged (every endpoint, every
 * effect); the shell is composed from layout regions (AppHeader, ActivityBar,
 * SidePanel, Inspector, BottomDock, StatusBar) and feature panels
 * (features/*) that receive their handlers as props — prompt_ui.txt §5.
 * Status messages go through i18n (`tt` = translator for callbacks).
 */
import { useState, useCallback, useEffect, useRef } from 'react'
import Viewport, { ViewportHandle, SegmentInstance, type CubeInfo } from './components/Viewport'
import type { UserVolume } from './components/assistantViz'
import { SyncPlayer } from './components/SyncPlayer'
import { BIMAnalysisPanel } from './components/BIMAnalysisPanel'
import TeamPanel from './components/TeamPanel'
import WebRTCCall from './components/WebRTCCall'
import BIMNavigator from './components/BIMNavigator'
import FuseScansModal from './components/FuseScansModal'
import CertifyKitPanel from './components/CertifyKitPanel'
import DeviationOverlay from './components/DeviationOverlay'
import type { IFCLoadResult } from './components/IFCLoader'
import { useAuth } from './context/AuthContext'
import LoginPage from './pages/LoginPage'
import AdminPage from './pages/AdminPage'
import SegmentationManager from './components/InteractiveSegmentation'
import { useConfirmDialog } from './components/ConfirmDialog'
import AssistantPanel from './components/AssistantPanel'
import {
  Search, Tag, Plug, Upload, Settings, Crosshair, Maximize, Monitor, Grid3X3, RotateCcw, Ruler, TriangleRight, Scissors,
  Move, BookOpen, Keyboard, Info, Users, LogOut, FolderOpen, Axis3D, Building2, Package, ArrowUpFromLine, Trash2, Unlock,
  Clock, Scale, BarChart3, Home, Camera, SlidersHorizontal, Sparkles, Undo2, Brush, Layers, Star, Terminal,
  ListTodo, FileCheck2, PanelLeftClose, PanelLeftOpen, Wand2,
  Circle, Sun, FlaskConical, Pin, Pentagon, Palette,
} from 'lucide-react'
import { useI18n, getT } from './i18n'
import { useLayout } from './layout/LayoutContext'
import { AppHeader, type HeaderMenu } from './layout/AppHeader'
import { ActivityBar, type ActivityItem } from './layout/ActivityBar'
import { SidePanel } from './layout/SidePanel'
import { Inspector } from './layout/Inspector'
import { BottomDock } from './layout/BottomDock'
import { StatusBar } from './layout/StatusBar'
import { CommandPalette, type Command } from './layout/CommandPalette'
import { registerCommands, findByName, type Executors, type Outcome } from './commands/registry'
import type { UiCommandId } from './commands/catalog.generated'
import { SettingsDialog } from './layout/SettingsDialog'
import { SceneLookDialog } from './layout/SceneLookDialog'
import { DEFAULT_LOOK, gridLength, parseLook, type SceneLook } from './components/viewport/sceneLook'
import { Tabs } from './components/ui/Tabs'
import { Button } from './components/ui/Button'
import { Banner } from './components/ui/Banner'
import { Dialog } from './components/ui/Dialog'
import { Field, SegmentedControl } from './components/ui/Field'
import { EmptyState } from './components/ui/EmptyState'
import { useToast } from './components/ui/Toast'
import type { MenuEntry } from './components/ui/Menu'
import type { ProgressStage } from './components/ui/Progress'
import { FloatingToolbar, type ColorMode } from './components/viewport/FloatingToolbar'
import { ViewportHud, type ReadoutAnchor, type ViewPreset } from './components/viewport/ViewportHud'
import { SplashOverlay } from './components/viewport/SplashOverlay'
import { WelcomeScreen } from './components/viewport/WelcomeScreen'
import { tokenColor } from './components/viewport/palette'
import { SessionsPanel } from './features/SessionsPanel'
import { InstancesPanel, type ListedRows } from './features/InstancesPanel'
import { ScansPanel } from './features/ScansPanel'
import { PipelineDialog } from './features/PipelineDialog'
import { ResumeDialog } from './features/ResumeDialog'
import { ObjectLibraryDialog } from './features/ObjectLibraryDialog'
import { AutosegmentDialog, type AutosegmentState, type AutosegmentStages } from './features/AutosegmentDialog'
import { GenreconDialog } from './features/GenreconDialog'
import { ReferenceDialog, type ReferenceMode } from './features/ReferenceDialog'
import { CalibrationDialog, type CalibrationDraft, type CalMeasurement, type CalPlane, type CalSummary } from './features/CalibrationDialog'
import { EvaluationPanel } from './features/EvaluationPanel'
import { PhysicsPanel } from './features/PhysicsPanel'
import { ConsolePanel, JobsPanel, PropertiesPanel } from './features/DockPanels'

/** Translator for callbacks defined before the i18n hook is read. */
const tt = (key: string, vars?: Record<string, string | number | null | undefined>) => getT()(key, vars)

interface SessionInfo {
  id: string
  name: string
  date: string
  frameCount: number
  chunkCount: number
  hasCloud: boolean
  hasSegments: boolean
  hasBim: boolean
  bimCount: number
  cloudSizeMb: number
  hasSabana: boolean
}

interface PipelineStageInfo {
  id: string
  label: string
  icon: string
  enabled: boolean
  status: string
  pct: number
  message: string
  elapsed: number
}

interface PipelineState {
  session_id?: string
  status: string
  current_stage_idx: number
  stages: PipelineStageInfo[]
  scans?: string[]  // scan keys being rebuilt
}

type Tool = 'navigate' | 'measure-distance' | 'measure-angle' | 'measure-plane' | 'section-box' | 'align' | 'erase'

const THREE_DEG = Math.PI / 180

/** stable identities for a bag of callbacks re-created on every render: each returned function calls
 *  the LATEST one, so a memoized child never sees a stale closure and never re-renders for a new
 *  function identity (USER 2026-10-16: the Instances list, re-rendered on every status update of the
 *  App, made the viewport crawl while the cloud moved) */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function useStableCallbacks<T extends Record<string, (...args: any[]) => any>>(fns: T): T {
  const latest = useRef<T>(fns)
  latest.current = fns
  const stable = useRef<T | null>(null)
  if (!stable.current) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const out: Record<string, (...args: any[]) => any> = {}
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    for (const k of Object.keys(fns)) out[k] = (...args: any[]) => (latest.current[k] as (...a: any[]) => any)(...args)
    stable.current = out as T
  }
  return stable.current
}

function App() {
  // auth first: token feeds the handlers' deps below (TDZ)
  const { user, token, loading: authLoading, logout } = useAuth()
  const { confirmDanger, dialogElement: appDialog } = useConfirmDialog()
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [activeSession, setActiveSession] = useState<string | null>(null)
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [activeTool, setActiveTool] = useState<Tool>('navigate')
  // reference distance / plane dialog of the evaluation battery (keyframe images); null = closed
  const [referenceDialog, setReferenceDialog] = useState<ReferenceMode | null>(null)
  // CALIBRATION ON THE 3D MODEL (USER 2026-11-13: "para calibrar debe hacerse tomando mediciones sobre el modelo 3d
  // no sobre las imagenes … se debe poder seleccionar que una medicion es de calibracion y deben quedar almacenadas"):
  // the pin of a measured distance opens the window with that measurement as its draft
  const [calibrationOpen, setCalibrationOpen] = useState(false)
  const [calibrationDraft, setCalibrationDraft] = useState<CalibrationDraft | null>(null)
  // the kind chosen in the Calibration button's selector (a distance or a plane): the next measurement of that
  // kind made on the model opens the window as its draft
  const [calibrating, setCalibrating] = useState<'distance' | 'plane' | null>(null)
  const calibratingRef = useRef<'distance' | 'plane' | null>(null)
  useEffect(() => { calibratingRef.current = calibrating }, [calibrating])
  // leaving the tool leaves the calibration it was armed for
  useEffect(() => {
    if (calibrating === 'distance' && activeTool !== 'measure-distance') setCalibrating(null)
    if (calibrating === 'plane' && activeTool !== 'measure-plane') setCalibrating(null)
  }, [activeTool, calibrating])
  const [eraseRadius, setEraseRadius] = useState(0.15)
  const [eraseMarks, setEraseMarks] = useState(0)
  const [eraseTarget, setEraseTarget] = useState<string>('')
  const [eraseShape, setEraseShape] = useState<'sphere' | 'cube' | 'box'>('sphere')
  const [eraseBoxSel, setEraseBoxSel] = useState(false)
  const [eraseYawDeg, setEraseYawDeg] = useState(0)
  // Brush confidence filter (user 2026-08-31): preview lights up the points
  // below the threshold in red; applying moves them to unsegmented.
  const [eraseConfThr, setEraseConfThr] = useState(0.25)
  const [eraseConfArmed, setEraseConfArmed] = useState(false)
  // Object library (user 2026-08-31): collection + every project GLB,
  // insertable as scene REFERENCES (delete never touches the source)
  const [showObjectLibrary, setShowObjectLibrary] = useState(false)
  const [objectLibrary, setObjectLibrary] = useState<Array<{ source: string; session?: string; name: string; url: string; size_mb: number }>>([])
  const [sceneObjSel, setSceneObjSel] = useState<number | null>(null)
  // gravity sandbox (USER 2026-10-01)
  const [physicsOpen, setPhysicsOpen] = useState(false)
  // shadows: meshes cast + receive, the cloud receives (USER 2026-10-01); on by default
  const [shadowsOn, setShadowsOn] = useState(true)
  useEffect(() => { viewportRef.current?.setShadows(shadowsOn) }, [shadowsOn])
  // points as cubes (USER 2026-10-04: "cubos como en la página de PointDiT"); OFF by default since
  // 2026-10-16 (USER: "destildado … para que siempre arranque con puntos")
  const [cubesOn, setCubesOn] = useState(false)
  useEffect(() => { viewportRef.current?.setPointShape(cubesOn ? 'cube' : 'disc') }, [cubesOn])
  // light intensity (USER 2026-10-16: "luz ambiente en la escena para que se vea mejor la nube" /
  // "¿se puede agregar intensidad de luz?"): the Display slider, 1 = the captured colour, saved in the prefs
  const [lightIntensity, setLightIntensity] = useState(1.0)
  useEffect(() => { viewportRef.current?.setAmbientLight(lightIntensity) }, [lightIntensity])
  const [pointOpacity, setPointOpacity] = useState(1.0)
  useEffect(() => { viewportRef.current?.setPointOpacity(pointOpacity) }, [pointOpacity])
  // the precision core's diagnostic layers (flyers, mono detail, mixed unresolved): off by default, per session
  const [flyersOn, setFlyersOn] = useState(false)
  const [monoDetailOn, setMonoDetailOn] = useState(false)
  const [unresolvedOn, setUnresolvedOn] = useState(false)
  useEffect(() => { void viewportRef.current?.setPrecisionLayer('flyers', flyersOn, activeSession) }, [flyersOn, activeSession])
  useEffect(() => { void viewportRef.current?.setPrecisionLayer('mono_detail', monoDetailOn, activeSession) }, [monoDetailOn, activeSession])
  useEffect(() => { void viewportRef.current?.setPrecisionLayer('mixed_unresolved', unresolvedOn, activeSession) }, [unresolvedOn, activeSession])
  const [physicsDiameterCm, setPhysicsDiameterCm] = useState(20)
  const [physicsBounce, setPhysicsBounce] = useState(0.7)
  const [physicsCount, setPhysicsCount] = useState(0)
  const [physicsPlacing, setPhysicsPlacing] = useState(false)
  useEffect(() => {                       // the ghost follows the panel's diameter / bounce while placing
    if (physicsPlacing) viewportRef.current?.physicsPlace(physicsDiameterCm / 100, physicsBounce)
    else viewportRef.current?.physicsStopPlacing()
  }, [physicsPlacing, physicsDiameterCm, physicsBounce])
  useEffect(() => {                       // Esc ends placing
    if (!physicsPlacing) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setPhysicsPlacing(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [physicsPlacing])
  useEffect(() => {                       // how many spheres are still in the scene (they fall out)
    if (!physicsOpen) return
    const id = window.setInterval(() => setPhysicsCount(viewportRef.current?.physicsCount() ?? 0), 500)
    return () => window.clearInterval(id)
  }, [physicsOpen])
  const [placedObjects, setPlacedObjects] = useState<Array<{ id: number; name: string; visible: boolean }>>([])
  const [alignTargets, setAlignTargets] = useState<Array<{ key: string; label: string }>>([])
  const [alignTarget, setAlignTarget] = useState('')
  //  Certification kit (claude_stac.txt §11): epochs before/after, colour by
  // witness status / mv_votes, trajectory with loop edges, duplicates,
  // attention list, acta — the user judges by selecting between the epochs.
  const [showCertifyKit, setShowCertifyKit] = useState(false)
  // bumped when a pipeline finishes: the certification stage ran inside it,
  // so an open kit re-reads the acta / epochs / edges
  const [certifyRefresh, setCertifyRefresh] = useState(0)
  // No button (USER 2026-09-13): the kit opens BY ITSELF when the session has
  // a certification acta with pending epochs to judge (after the pipeline's
  // certify stage, or on opening such a session).
  useEffect(() => {
    if (!activeSession) { setShowCertifyKit(false); return }
    let cancelled = false
    fetch(`/api/certify/state/${activeSession}`)
      .then(r => (r.ok ? r.json() : null))
      .then(st => {
        if (cancelled || !st) return
        if (st.acta && (st.epochs?.length ?? 0) > 1) setShowCertifyKit(true)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [activeSession, certifyRefresh])
  // ── Multi-scan project, VS-Code-style TABS (USER 2026-09-06): one tab
  // per open scan; the ACTIVE tab is the scan being viewed AND worked on
  // (segmentation, brush, meshing, chat). Closing a tab loses
  // nothing; the scans list reopens it (double-click). The composition
  // reference is a  on the list row (one per project).
  type ScanTab = { key: string; label: string; date: string; kind: 'scan' | 'fused' }
  const [projectScans, setProjectScans] = useState<any[]>([])
  // scans of EVERY project (tree children in the Projects list); the
  // loaded project's entry is refreshed together with projectScans
  const [allProjectScans, setAllProjectScans] = useState<Record<string, any[]>>({})
  const [scanTabs, setScanTabs] = useState<ScanTab[]>([])
  const [activeScanTab, setActiveScanTab] = useState<string | null>(null)
  // project whose scans are being fused — opened from the project row of the
  // Sessions tab, so it is not necessarily the loaded one (USER 2026-09-24)
  const [fuseSession, setFuseSession] = useState<string | null>(null)
  const openFuse = useCallback((sid: string) => setFuseSession(sid), [])
  // scan requested for the NEXT project open (tree double-click on a
  // not-loaded project); null  the composition reference opens
  const [openScanKey, setOpenScanKey] = useState<string | null>(null)
  const openScanKeyRef = useRef<string | null>(null)
  useEffect(() => { openScanKeyRef.current = openScanKey }, [openScanKey])
  const activeScanTabRef = useRef<string | null>(null)
  useEffect(() => { activeScanTabRef.current = activeScanTab }, [activeScanTab])
  const loadProjectScans = useCallback(async (sid: string): Promise<any[]> => {
    try {
      const r = await fetch(`/sessions/${sid}/scans`)
      if (!r.ok) return []
      const d = await r.json()
      const scans: any[] = d.scans || []
      setProjectScans(scans)
      setAllProjectScans(prev => ({ ...prev, [sid]: scans }))
      return scans
    } catch { return [] }
  }, [])
  // tree children for projects that are NOT loaded (best effort, parallel)
  const loadAllProjectScans = useCallback(async (ids: string[]) => {
    const entries: Array<[string, any[]]> = await Promise.all(ids.map(async (id): Promise<[string, any[]]> => {
      try {
        const r = await fetch(`/sessions/${id}/scans`)
        if (!r.ok) return [id, []]
        const d = await r.json()
        const scans: any[] = (d.scans || []).filter((sc: any) => sc.key !== 'legacy/default')
        return [id, scans]
      } catch { return [id, []] }
    }))
    setAllProjectScans(prev => {
      const next = { ...prev }
      for (const [id, scans] of entries) next[id] = scans
      return next
    })
  }, [])
  // switch the backend's active scan and reload the main cloud from it
  const activateScan = useCallback(async (sid: string, sc: { key: string; label: string; date: string; kind: 'scan' | 'fused' }, reload = true) => {
    setScanTabs(prev => prev.some(t => t.key === sc.key) ? prev : [...prev, { key: sc.key, label: sc.label, date: sc.date, kind: sc.kind }])
    setActiveScanTab(sc.key)
    try {
      await fetch(`/api/project/${sid}/active_scan`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scan_key: sc.key }),
      })
    } catch { /* legacy single-scan sessions have no project */ }
    if (reload) {
      viewportRef.current?.clearScanLayers()
      viewportRef.current?.sendCommandPreserveCamera({ type: 'load_session', session_id: sid, scan_key: sc.key })
      setStatusMessage(tt('scans.workingOn', { label: sc.label }))
    }
  }, [])
  const closeScanTab = useCallback((key: string) => {
    setScanTabs(prev => {
      const idx = prev.findIndex(t => t.key === key)
      const next = prev.filter(t => t.key !== key)
      if (activeScanTabRef.current === key && activeSession) {
        const neighbour = next[Math.max(0, idx - 1)]
        if (neighbour) activateScan(activeSession, neighbour)
        else setActiveScanTab(null)
      }
      return next
    })
  }, [activeSession, activateScan])
  // on session load: fetch the scans and open the active (or first) one as a tab
  useEffect(() => {
    viewportRef.current?.clearScanLayers()
    setProjectScans([]); setScanTabs([]); setActiveScanTab(null)
    if (!activeSession) return
    loadProjectScans(activeSession).then(scans => {
      if (!scans.length) return
      // opening a project shows the composition REFERENCE () — the
      // backend resets the active scan to it on the open load — unless a
      // scan was asked for explicitly from the tree
      const want = openScanKeyRef.current
      openScanKeyRef.current = null
      setOpenScanKey(null)
      const cur = (want && scans.find(s => s.key === want))
        || scans.find(s => s.is_reference) || scans.find(s => s.is_active) || scans[0]
      // the session load already showed this scan — register the tab only
      activateScan(activeSession, { key: cur.key, label: cur.label, date: cur.date, kind: cur.kind || 'scan' }, false)
    })
  }, [activeSession, loadProjectScans, activateScan])
  // Ctrl+Tab cycles tabs
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey && e.key === 'Tab') || scanTabs.length < 2 || !activeSession) return
      e.preventDefault()
      const i = scanTabs.findIndex(t => t.key === activeScanTabRef.current)
      const nxt = scanTabs[(i + (e.shiftKey ? scanTabs.length - 1 : 1)) % scanTabs.length]
      activateScan(activeSession, nxt)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [scanTabs, activeSession, activateScan])

  // Resume/Cancel dialog for incomplete propagation (USER 2026-09-06): on
  // opening the Segmentation Manager, if any instance was not fully
  // propagated (a prior pass OOM'd), ask before opening.
  const [resumeDialog, setResumeDialog] = useState<{ session: string; incomplete: any[]; complete: number } | null>(null)
  const [resumeBusy, setResumeBusy] = useState(false)
  // live progress of the resume task (USER 2026-09-06: "por dónde va y
  // cuánto falta" — the modal stays open until the task finishes)
  const [resumeProgress, setResumeProgress] = useState<{ pct: number; detail: string; elapsed_s: number } | null>(null)
  useEffect(() => {
    if (!resumeBusy || !resumeDialog) return
    const sid = resumeDialog.session
    const iv = setInterval(async () => {
      try {
        const d = await fetch(`/api/tasks/${sid}`).then(r => r.json())
        const t = (d.tasks || []).find((x: any) => x.task_type === 'resume')
        if (t) setResumeProgress({ pct: t.pct || 0, detail: t.detail || '', elapsed_s: t.elapsed_s || 0 })
      } catch { /* keep last */ }
    }, 2000)
    return () => clearInterval(iv)
  }, [resumeBusy, resumeDialog])
  // Open the manager directly (it loads SAM3). The Resume/Cancel dialog is
  // raised AFTER SAM3 reaches "ready" (USER 2026-09-06: Cancel just clears
  // the incomplete ones and you keep segmenting; Resume finds SAM3 already
  // loaded). The check-on-ready lives in a dedicated effect below.
  const [resumeCheckedFor, setResumeCheckedFor] = useState<string | null>(null)
  const openSegmentationManager = useCallback((sid: string) => {
    setResumeCheckedFor(null)
    setInteractiveSessionId(sid)
  }, [])
  // leaving the brush turns the red preview off
  useEffect(() => {
    if (activeTool !== 'erase') {
      setEraseConfArmed(false)
      viewportRef.current?.setConfHighlight(null)
    }
  }, [activeTool])
  // shell state now lives in LayoutContext (prompt_ui.txt §5)
  const layoutCtx = useLayout()
  const setActivePanel = useCallback((p: 'sessions' | 'segments' | 'bim' | 'team' | 'analysis' | 'assistant' | null) => {
    if (p === null) layoutCtx.setLeftTab(null)
    else if (p === 'segments') layoutCtx.setLeftTab('instances')
    else if (p === 'analysis') layoutCtx.openDock('report')
    else if (p === 'assistant') layoutCtx.openInspector('assistant')
    else layoutCtx.setLeftTab(p)
  }, [layoutCtx])
  const consoleOpen = layoutCtx.state.dockOpen && layoutCtx.state.dockTab === 'console'
  const setConsoleOpen = useCallback((open: boolean) => { if (open) layoutCtx.openDock('console'); else layoutCtx.setDockOpen(false) }, [layoutCtx])
  const [connected, setConnected] = useState(false)
  const [serverAlive, setServerAlive] = useState(false)

  const [bimModels, setBimModels] = useState<IFCLoadResult[]>([])
  const [vlmStatus, setVlmStatus] = useState<'up' | 'loading' | 'busy' | 'down' | null>(null)
  const [pointSize, setPointSize] = useState(5.0)
  // Potree LOD point budget (max points rendered at once). Higher = more of the
  // cloud visible, but more client GPU/RAM. Slider lets it scale to any
  // cloud/machine. Default 10M (good overview for large clouds; raise if your
  // viewer GPU handles it).
  const [pointBudget, setPointBudget] = useState(10_000_000)
  // Display-settings popover (groups Point Size / Detail / Confidence so they
  // don't eat toolbar space).
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.0)
  const [hasConfidence, setHasConfidence] = useState(false)

  const [showAxes, setShowAxes] = useState(false)
  const [showGrid, setShowGrid] = useState(true)
  // the session's background and grid (USER 2026-11-09: View, Background and grid), saved in its prefs
  const [sceneLook, setSceneLook] = useState<SceneLook>(DEFAULT_LOOK)
  const [sceneLookOpen, setSceneLookOpen] = useState(false)
  const [pointCount, setPointCount] = useState(0)
  // STICKY: the viewer HAS an octree for this session. `pointCount` is the
  // VISIBLE count, reported every frame — it dips to 0 whenever Potree swaps
  // loaders (a new cloud pushed mid-pipeline: CloudCompy's, then the octree)
  // and the covering overlay used to slam back over a cloud the user was
  // already looking at. USER 2026-09-23: *"si hay octree enviado al ui, no me
  // lo pise con el panel azul de reconstruccion ni ningun otro panel azul"*.
  // Cleared only where the scene really goes away (disconnect / session switch
  // / clear / logout), never by a frame that happened to render nothing.
  const [cloudEverShown, setCloudEverShown] = useState(false)
  const [sessionLoading, setSessionLoading] = useState<string | null>(null)
  const [fps, setFps] = useState(0)
  const [consoleLogs, setConsoleLogs] = useState<{ ts: string; level: string; msg: string }[]>([])
  const viewportRef = useRef<ViewportHandle>(null)
  // Timestamp (ms) of the last time a cloud actually loaded into the viewer (point
  // count > 0). Used by the pipeline-done handler to tell "the cloud arrived on this
  // live socket" from "WS reconnected mid-pipeline  potree_ready went to a dead socket".
  const lastCloudLoadAtRef = useRef(0)
  const activeSessionRef = useRef<string | null>(null)
  useEffect(() => { activeSessionRef.current = activeSession }, [activeSession])
  const [flythroughOpen, setFlythroughOpen] = useState<string | null>(null)
  const [adminOpen, setAdminOpen] = useState(false)

  // ── Sábana state ──
  const [sabanaVisible, setSabanaVisible] = useState(false)
  const [sabanaLoading, setSabanaLoading] = useState(false)
  const [sabanaMetrics, setSabanaMetrics] = useState<any>(null)
  const [sabanaFullMeta, setSabanaFullMeta] = useState<any>(null)

  // ── Camera Poses state ──
  const [showCameraPoses, setShowCameraPoses] = useState(true)
  const [hasCameraPoses, setHasCameraPoses] = useState(false)

  // Team / WebRTC state
  const teamWsRef = useRef<WebSocket | null>(null)
  const [callTarget, setCallTarget] = useState<{ userId: number; username: string } | null>(null)
  const [incomingCall, setIncomingCall] = useState<{ from: number; username: string; callId: string; media: string } | null>(null)

  // Pipeline state
  const [pipelineDialogOpen, setPipelineDialogOpen] = useState(false)
  const [pipelineDialogSession, setPipelineDialogSession] = useState<string | null>(null)

  // The pipeline always runs end-to-end (reconstruction  cloud cleaning 
  // TSDF); the backend ignores any stage selection, so the dialog only asks
  // which scans to rebuild and whether to replace existing outputs.
  const [pipelineReplace, setPipelineReplace] = useState(true)
  // scans whose segmentation chain runs after the cloud — OFF by default (USER 2026-10-05)
  const [pipelineSegment, setPipelineSegment] = useState<string[]>([])
  const [autosegOpen, setAutosegOpen] = useState(false)
  const [autosegState, setAutosegState] = useState<AutosegmentState | null>(null)
  const [autosegLoading, setAutosegLoading] = useState(false)
  const [autosegScans, setAutosegScans] = useState<string[]>([])
  const [pipelineRunning, setPipelineRunning] = useState<PipelineState | null>(() => {
    try {
      const saved = sessionStorage.getItem('pipelineRunning')
      return saved ? JSON.parse(saved) : null
    } catch { return null }
  })
  const pipelineRunningRef = useRef<PipelineState | null>(pipelineRunning)
  // RECONCILE AGAINST THE SERVER, never trust the restored state.
  // `sessionStorage` is a convenience so a reload does not lose a running
  // pipeline; it is not the truth. On 2026-09-21 a pipeline finished while
  // the viewer was disconnected, the final event reached nobody, and every
  // reload restored "running at 5%" from storage — the blue panel covered the
  // cloud forever although the server had said `pipelines: {}` all along.
  // The same reasoning applies to the epoch and the octree below: state is
  // asked for, not caught.
  // EVERY active pipeline, not just the one this tab launched (USER 2026-09-23:
  // "los proyectos que mande deben identificar cual esta en reconstruccion,
  // cual en cola"). The manager runs one at a time and queues the rest, so a
  // project list has to say which is which. Polled, because the progress
  // broadcast only carries the job that moved.
  const [pipelineJobs, setPipelineJobs] = useState<Record<string, { status: string; queue_position: number; scan_key?: string }>>({})
  useEffect(() => {
    let stop = false
    const poll = async () => {
      try {
        const r = await fetch('/api/pipelines/active')
        if (r.ok && !stop) {
          const live = ((await r.json())?.pipelines ?? {}) as Record<string, any>
          // which SCAN the job is on: the manager resolves it into session_dir
          // (".../scans/<date>/src_<source>"), and the tree keys its scans
          // "<date>/<source>" — so the row that is actually being rebuilt can
          // be marked (USER 2026-09-23, multi-scan projects).
          const scanKeyOf = (dir?: string): string | undefined => {
            const m = /\/scans\/([^/]+)\/src_([^/]+)/.exec(String(dir ?? ''))
            return m ? `${m[1]}/${m[2]}` : undefined
          }
          // one job per session + scan since 2026-10-05 (every scan of a multi-scan
          // order waits in the queue as its own job): group them by session — the
          // project row shows the best state, each scan row its own
          const grouped: Record<string, { status: string; queue_position: number; scan_key?: string; scans: Record<string, { status: string; queue_position: number }> }> = {}
          for (const j of Object.values(live)) {
            const sid = String(j.session_id ?? '')
            if (!sid) continue
            const st = String(j.status)
            const qp = Number(j.queue_position ?? 0)
            const sk = (j.scan_key ? String(j.scan_key) : undefined) ?? scanKeyOf(j.session_dir)
            const g = grouped[sid] ?? (grouped[sid] = { status: st, queue_position: qp, scan_key: sk, scans: {} })
            if (sk) g.scans[sk] = { status: st, queue_position: qp }
            if (st === 'running' || (g.status !== 'running' && qp < g.queue_position)) {
              g.status = st; g.queue_position = qp; g.scan_key = sk
            }
          }
          setPipelineJobs(grouped)
          // USER 2026-11-05 ("cuando terminó no le avisó al visor … tuve que refrescar la página"): a job the server
          // no longer lists has ENDED even when its final event never reached this tab — the state restored or
          // caught here is cleared at once (it used to wait for a reload or a window focus)
          setPipelineRunning(prev => (prev && !Object.values(live).some(j => String(j.session_id ?? '') === prev.session_id))
            ? null : prev)
        }
      } catch { /* offline: keep the last picture rather than blank the list */ }
      if (!stop) window.setTimeout(poll, 4000)
    }
    void poll()
    return () => { stop = true }
  }, [])

  useEffect(() => {
    let cancelled = false
    const reconcile = async () => {
      try {
        const r = await fetch('/api/pipelines/active')
        if (!r.ok || cancelled) return
        const live = ((await r.json())?.pipelines ?? {}) as Record<string, unknown>
        setPipelineRunning(prev => {
          if (!prev) return prev
          if (Object.values(live).some(j => (j as { session_id?: string }).session_id === prev.session_id)) return prev
          // the server does not know it: it finished while we were away
          return null
        })
      } catch { /* offline: keep what we have rather than guess */ }
    }
    reconcile()
    const onFocus = () => { void reconcile() }
    window.addEventListener('focus', onFocus)
    return () => { cancelled = true; window.removeEventListener('focus', onFocus) }
  }, [])
  useEffect(() => {
    pipelineRunningRef.current = pipelineRunning
    if (pipelineRunning && pipelineRunning.status !== 'done' && pipelineRunning.status !== 'failed' && pipelineRunning.status !== 'cancelled') {
      sessionStorage.setItem('pipelineRunning', JSON.stringify(pipelineRunning))
    } else {
      sessionStorage.removeItem('pipelineRunning')
    }
  }, [pipelineRunning])
  const [interactiveSessionId, setInteractiveSessionId] = useState<string | null>(null)
  const interactiveSessionRef = useRef(interactiveSessionId)
  const [compareDialogOpen, setCompareDialogOpen] = useState(false)
  const [useManualAlignment, setUseManualAlignment] = useState(true)
  // Video upload  frame extraction, keyed by session id. Presence means the
  // session has no frames yet and is uploading/extracting; pct drives the spinner.
  const [extractingSessions, setExtractingSessions] = useState<Record<string, number>>({})
  const videoInputRef = useRef<HTMLInputElement>(null)
  const pendingVideoSession = useRef<string | null>(null)
  const [scansList, setScansList] = useState<{ date: string; source: string; key: string; frame_count: number; has_output: boolean; recon_state?: string; cached_chunks?: number }[]>([])
  const [selectedScans, setSelectedScans] = useState<string[]>([])




  // Periodic server health check — updates indicator only
  // NOTE: timeout/threshold are generous because the Python server does heavy
  // CPU-bound work (BIM registration, SAM3 loading, reconstruction) that blocks
  // the GIL for up to several minutes. Short thresholds cause false disconnects.
  const failCountRef = useRef(0)
  const FAIL_THRESHOLD = 10 // ~60s of unresponsiveness before declaring dead
  useEffect(() => {
    interactiveSessionRef.current = interactiveSessionId
  }, [interactiveSessionId])
  // Raise the Resume/Cancel dialog AFTER SAM3 reaches "ready" for the just-
  // opened manager (USER 2026-09-06): Cancel clears the incomplete ones and
  // you keep segmenting; Resume finds SAM3 already loaded.
  useEffect(() => {
    const sid = interactiveSessionId
    if (!sid || resumeCheckedFor === sid) return
    let cancelled = false
    const check = async () => {
      for (let i = 0; i < 150 && !cancelled; i++) {
        const s = await fetch(`/api/segmentation/init_status/${sid}`).then(r => r.json()).catch(() => ({}))
        if (s.status === 'ready') break
        if (s.status === 'error') return
        await new Promise(r => setTimeout(r, 2000))
      }
      if (cancelled) return
      try {
        const st = await fetch(`/api/segmentation/resume/status/${sid}`).then(r => r.json())
        if (!cancelled && st.needs_resume) {
          setResumeDialog({ session: sid, incomplete: st.incomplete || [], complete: st.complete || 0 })
        }
      } catch { /* silent */ }
      if (!cancelled) setResumeCheckedFor(sid)
    }
    check()
    return () => { cancelled = true }
  }, [interactiveSessionId, resumeCheckedFor])
  // Health poll: 5 s while the server answers, BACKING OFF to 30 s once it is
  // declared down. A dead backend used to be polled forever at 5-10 s by this
  // loop and by the login page at the same time — pure noise in the proxy log
  // and in the server's own access log, and the status bar already says it is
  // down. The first failure still reacts at full speed.
  useEffect(() => {
    let intervalMs = 5000
    let timerId: ReturnType<typeof setTimeout> | null = null
    const downDelay = () =>
      Math.min(30000, 5000 * 2 ** Math.max(0, failCountRef.current - FAIL_THRESHOLD))
    const checkHealth = async () => {
      try {
        const resp = await fetch('/health', { signal: AbortSignal.timeout(60000) })
        if (resp.ok) {
          failCountRef.current = 0
          setServerAlive(true)
          intervalMs = 5000
        } else {
          failCountRef.current++
          if (failCountRef.current >= FAIL_THRESHOLD) setServerAlive(false)
          intervalMs = failCountRef.current >= FAIL_THRESHOLD ? downDelay() : 5000
        }
      } catch {
        failCountRef.current++
        if (failCountRef.current >= FAIL_THRESHOLD) setServerAlive(false)
        intervalMs = failCountRef.current >= FAIL_THRESHOLD ? downDelay() : 5000
      }
      timerId = setTimeout(checkHealth, intervalMs)
    }
    checkHealth()
    return () => { if (timerId) clearTimeout(timerId) }
  }, [])

  // When server goes down, clear session state
  const prevAliveRef = useRef(serverAlive)
  useEffect(() => {
    if (prevAliveRef.current && !serverAlive) {
      // Server just went from alive  dead (after 3 consecutive failures)
      setConnected(false)
      setActivePanel('sessions')
      setSessions([])
      setActiveSession(null)
      setSelectedSession(null)
      setSegments([])
      setBimModels([])
      setPointCount(0)
      setCloudEverShown(false)
      setPipelineRunning(null)
      setSessionLoading(null)
      setStatusMessage(tt('status.serverDisconnected'))
      // Dispose Potree data from GPU to free memory and stop LOD updates
      viewportRef.current?.clearScene()
    }
    if (!prevAliveRef.current && serverAlive) {
      // Server just went from dead  alive: clear stale message
      setStatusMessage('')
    }
    prevAliveRef.current = serverAlive
  }, [serverAlive])

  // Console log WebSocket
  useEffect(() => {
    if (!consoleOpen) return
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/logs`)
    ws.onmessage = (ev) => {
      try {
        const entry = JSON.parse(ev.data)
        setConsoleLogs(prev => {
          const next = [...prev, entry]
          return next.length > 500 ? next.slice(-400) : next
        })
      } catch { /* ignore */ }
    }
    ws.onclose = () => { }
    return () => ws.close()
  }, [consoleOpen])

  const [statusMessage, setStatusMessage] = useState('')
  const [segments, setSegments] = useState<SegmentInstance[]>([])
  // Unsegmented point count for the panel (user 2026-08-31: every segment
  // shows its count except Unsegmented). null = unknown  hidden.
  const [unsegmentedCount, setUnsegmentedCount] = useState<number | null>(null)
  // Masks that are NOT objects: the matching resolved them into another
  // instance (same space, or same label and contiguous) or they matched no
  // cloud point. They are kept out of the list — but the panel says how many,
  // so the fusion is visible instead of silent (USER 2026-09-17).
  const [absorbedCount, setAbsorbedCount] = useState<number | null>(null)
  const refreshUnsegmentedCount = useCallback(async (sid: string | null) => {
    if (!sid) { setUnsegmentedCount(null); return }
    try {
      const r = await fetch(`/api/sessions/${sid}/segmentation`)
      if (r.ok) {
        const d = await r.json()
        setUnsegmentedCount(typeof d.unsegmented_points === 'number' ? d.unsegmented_points : null)
        setAbsorbedCount(Array.isArray(d.absorbed) ? d.absorbed.length : null)
      }
    } catch { /* silent */ }
  }, [])
  // Reload the segment list from the server PRESERVING the user's visibility
  // choices (user 2026-08-31: refreshes that reset everything to visible
  // desynced the panel from the viewer — and the panel's visible-list is the
  // brush's protection list, so the desync corrupted transactions).
  const reloadSegments = useCallback(async (sid: string | null) => {
    if (!sid) return
    try {
      const r = await fetch(`/api/sessions/${sid}/segmentation`)
      if (!r.ok) return
      const d = await r.json()
      if (typeof d.unsegmented_points === 'number') setUnsegmentedCount(d.unsegmented_points)
      setAbsorbedCount(Array.isArray(d.absorbed) ? d.absorbed.length : null)
      if (Array.isArray(d.instances)) {
        setSegments(prev => {
          const vis = new Map(prev.map(s => [s.id, s.visible]))
          return d.instances.map((inst: any) => ({
            key: inst.global_id || `${inst.label}_${inst.instance_id || inst.id}`,
            id: inst.instance_id || inst.id,
            // the CLASS BYTE the octree carries (compact above 255 instances) — every
            // visibility call is keyed by it; a list rebuilt without it toggled nothing
            // or the wrong object (pccr 2026-10-01)
            classId: (inst.class_byte ?? (inst.instance_id || inst.id)) as number,
            label: `${inst.label}`,
            color: inst.color || tokenColor('--measure'),
            totalPoints: inst.total_points || 0,
            visible: vis.get(inst.instance_id || inst.id) ?? true,
            excluded: inst.excluded || false,
          }))
        })
      }
    } catch { /* silent */ }
  }, [])
  // Floory=0 leveling: candidates + which floor instance sits at y=0.
  // Changing the combobox re-levels instantly (no confirm); the manager
  // close and session load call the auto modes.
  const [floorLevel, setFloorLevel] = useState<{ candidates: { instance_id: number, label: string, height_m: number | null }[], selected: number | null }>({ candidates: [], selected: null })
  const floorLevelCheckedRef = useRef<string | null>(null)

  // Cloud object-level visibility — App owns the truth (Unsegmented + segments).
  // Reset to "visible" on every session switch; the second useEffect combines
  // it with `segments` and tells the Viewport whether to keep the cloud in the
  // scene (true) or pull it out (false  frees GPU, the Potree LOD self-gates,
  // and the raycaster skips it so measurements land on meshes).
  const [unsegmentedVisible, setUnsegmentedVisible] = useState(true)
  useEffect(() => {
    setUnsegmentedVisible(true)
    setUnsegmentedCount(null)
    refreshUnsegmentedCount(activeSession)
    // running flags are per-job, progress is per-session: switching sessions
    // must not leave the Mesh buttons stuck on a job from another session
  }, [activeSession, refreshUnsegmentedCount])
  useEffect(() => {
    if (!activeSession) return
    const anyVisible = unsegmentedVisible || segments.some(s => s.visible)
    viewportRef.current?.setCloudObjectVisible(anyVisible)
  }, [activeSession, unsegmentedVisible, segments])

  // Sidebar lists for generated meshes (visibility toggles only — generation
  // happens through the modals). Folder is the on-disk dir name used as a
  // stable React key when instance_id collides between shape and tsdf.
  type MeshListItem = {
    instanceId: number
    label: string
    folder: string
    visible: boolean
  }
  const [tsdfMeshes, setTsdfMeshes] = useState<MeshListItem[]>([])

  // Same for TSDF mesh list.
  // `resetVisibility` (default true) follows a viewer reset (reloadTsdf / session load) that drops
  // every per-instance mesh; false = an in-place refresh (USER 2026-10-15: a new mesh folder from any
  // writer joins the list while the shown meshes keep their state)
  const refreshTsdfMeshList = useCallback(async (sessionId: string, resetVisibility: boolean = true) => {
    try {
      const res = await fetch(`/api/segmentation/tsdf/list/${sessionId}`)
      if (!res.ok) { setTsdfMeshes([]); return }
      const data = await res.json()
      const items = (data?.shapes || []) as Array<{
        folder: string
        meta: { instance_id?: number; label?: string; method?: string }
      }>
      setTsdfMeshes(prev => {
        const prevVis = new Map(prev.map(m => [m.folder, m.visible]))
        // Include every TSDF mesh — per-object AND the whole-scene mesh
        // (folder "scene", which has no instance_id).
        return items.map(s => ({
          instanceId: s.meta?.instance_id ?? -1,
          label: (s.meta?.label as string)
            || (s.meta?.method === 'tsdf_scene' ? ' TSDF — whole scene'
                : s.meta?.method === 'poisson_scene' ? ' Poisson — whole scene'
                : s.folder === 'scene' ? ' Whole scene' : s.folder),
          folder: s.folder,
          // EVERY mesh starts hidden and not loaded (USER 2026-11-13: the GenRecon meshes off when a
          // session opens, so it opens fast): a refresh that resets visibility follows a viewer reset
          // (reloadTsdf / session load) that left them all pending, so every checkbox is unchecked —
          // checking it triggers the on-demand load in the Viewport (setTsdfVisibility).
          // A mesh that is NEW in a refresh that keeps visibility (tsdf_ready, the refresh button)
          // was just loaded into the scene by the Viewport (addNewTsdfEntries) and shows checked
          // (USER 2026-10-15).
          visible: resetVisibility ? false : (prevVis.get(s.folder) ?? true),
        }))
      })
    } catch { /* ignore */ }
  }, [])

  // Refresh the TSDF mesh list when the active session changes (or clear
  // them on session unload). The Viewport already auto-loads the GLBs into
  // the scene; this keeps the sidebar list in sync with what's on disk.
  useEffect(() => {
    if (!activeSession) {
      setTsdfMeshes([])
      return
    }
    refreshTsdfMeshList(activeSession)
  }, [activeSession, refreshTsdfMeshList])

  const connectToServer = useCallback(async () => {
    try {
      const headers: HeadersInit = {}
      if (token) headers['Authorization'] = `Bearer ${token}`
      const res = await fetch('/sessions', { headers })
      const data = await res.json()
      const list = Array.isArray(data) ? data : (data.sessions || [])
      const sessionList: SessionInfo[] = list.map((s: any) => ({
        id: s.id || s.session_id,
        name: s.id || s.session_id,
        date: s.date || s.created || '',
        frameCount: s.frame_count || 0,
        chunkCount: s.chunk_count || 0,
        hasCloud: s.has_cloud || false,
        hasSegments: s.has_segments || false,
        hasBim: s.has_bim || false,
        bimCount: s.bim_count || 0,
        cloudSizeMb: s.cloud_size_mb || 0,
        hasSabana: s.has_sabana || false,
      }))
      setSessions(sessionList)
      setConnected(true)
      loadAllProjectScans(sessionList.map(s => s.id))
      // console.log(`[STAC] Connected. ${sessionList.length} sessions found.`)
    } catch (err) {
      console.error('[STAC] Failed to connect:', err)
      setConnected(false)
      setActivePanel('sessions')
    }
  }, [token, loadAllProjectScans])

  // Recover active pipeline state after page load/refresh
  useEffect(() => {
    if (!connected) return
    fetch('/api/pipelines/active')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (!data?.pipelines) {
          // Server restarted with no active pipelines — clear stale state
          setPipelineRunning(null)
          return
        }
        const entries = Object.entries(data.pipelines)
        if (entries.length > 0) {
          // Restore the first running pipeline
          const [, job] = entries[0] as [string, any]
          setPipelineRunning({
            session_id: job.session_id,
            status: job.status,
            current_stage_idx: job.current_stage_idx,
            stages: job.stages,
          })
        } else {
          // No active pipelines — clear stale state
          setPipelineRunning(null)
        }
      })
      .catch(() => { })
  }, [connected])

  // Select session — highlight only, no cloud loading
  const handleSessionSelect = useCallback((sessionId: string) => {
    setSelectedSession(sessionId)
  }, [])

  // Projects-tree double-click on a scan: on the loaded project it opens the
  // tab; on another project it makes that scan the active one and loads the
  // project (the load then registers the tab).
  const openScanFromTree = useCallback(async (sid: string, sc: any) => {
    const tab = { key: sc.key, label: sc.label, date: sc.date, kind: (sc.kind || 'scan') as 'scan' | 'fused' }
    if (sid === activeSession) { activateScan(sid, tab); return }
    try {
      await fetch(`/api/project/${sid}/active_scan`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scan_key: sc.key }),
      })
    } catch { /* legacy */ }
    openScanKeyRef.current = sc.key
    setOpenScanKey(sc.key)     // the open load asks for this scan explicitly
    handleSessionLoad(sid)
  }, [activeSession, activateScan])  

  // Load session — actually loads the cloud in the viewport
  const handleSessionLoad = useCallback((sessionId: string) => {
    // Check if session has a cloud to load
    const sess = sessions.find(s => s.id === sessionId)
    if (sess && !sess.hasCloud && sess.chunkCount === 0) {
      // Nothing loadable yet — route straight to the reconstruction dialog.
      // It auto-detects cached/partial runs and defaults to resume mode.
      handleReconstruct(sessionId)
      return
    }
    // Unload current session if any
    viewportRef.current?.clearScene()
    viewportRef.current?.sendCommand({ type: 'cleared' })
    // Clear React state
    setSegments([])
    setBimModels([])
    setPointCount(0)
    setCloudEverShown(false)
    setStatusMessage('')
    setSessionLoading(sessionId)
    setActiveSession(sessionId)
    setSelectedSession(sessionId)
    setActiveTool('navigate')
    // Clear sábana state
    setSabanaVisible(false)
    setSabanaMetrics(null)
    setSabanaFullMeta(null)
    // Reset OBBs visibility
    viewportRef.current?.setOBBsVisible(true)
  }, [sessions])  

  // Dismiss loading overlay when points arrive (session load only, not refresh). USER 2026-11-05 (zaragoza, an IFC
  // that never finished loading: the toolbar never appeared): the cloud on screen ends the load — a session's BIM
  // and its deviation map keep loading behind it with their own status (the sábana check below still runs)
  const prevPointCount = useRef(0)
  useEffect(() => {
    if (prevPointCount.current === 0 && pointCount > 0 && sessionLoading) {
      setSessionLoading(null)
    }
    prevPointCount.current = pointCount
  }, [pointCount, sessionLoading, sessions, activeSession])

  // Timeout: if loading overlay stays for 15s without points, dismiss with error
  useEffect(() => {
    if (!sessionLoading) return
    const timer = setTimeout(() => {
      if (pointCount === 0 && sessionLoading) {
        setSessionLoading(null)
        // Only show error if no pipeline is running on this session
        if (!pipelineRunning || pipelineRunning.status !== 'running' || pipelineRunning.session_id !== activeSession) {
          setStatusMessage(tt('sessions.noCloudData'))
        }
      }
    }, 45000)  // 45s: large octrees need time to download before points appear
    return () => clearTimeout(timer)
  }, [sessionLoading, pipelineRunning, activeSession])  

  // Auto-detect sábana when BIM models first appear after session load
  const prevBimCount = useRef(0)
  useEffect(() => {
    const wasZero = prevBimCount.current === 0
    prevBimCount.current = bimModels.length
    // Only trigger when transitioning from 0  N models (fresh session load)
    if (!wasZero || bimModels.length === 0 || !activeSession) return
    fetch(`/api/sessions/${activeSession}/sabana/exists`)
      .then(r => r.json())
      .then(d => {
        if (d.exists) {
          setSabanaVisible(true)
          setSabanaLoading(true)
          setSessionLoading(tt('deviation.loadingComparison'))
          setStatusMessage(tt('deviation.loadingComparison'))
          viewportRef.current?.sendCommand({ type: 'load_sabana', session_id: activeSession })
          viewportRef.current?.setOBBsVisible(false)
          fetch(`/api/sessions/${activeSession}/sabana/meta`)
            .then(r2 => r2.ok ? r2.json() : null)
            .then(fullMeta => {
              if (fullMeta) {
                setSabanaMetrics(fullMeta)
                setSabanaFullMeta(fullMeta)
                setActivePanel('analysis')
              }
            })
            .catch(() => { })
        } else {
          setActivePanel('bim')
          setSessionLoading(null)
        }
      })
      .catch(() => { setActivePanel('bim'); setSessionLoading(null) })
  }, [bimModels, activeSession])

  // the overlay text the tasks poll put up, so it takes down exactly that one
  const taskOverlayRef = useRef<string | null>(null)
  // Poll server for active tasks (survives reconnect)
  useEffect(() => {
    if (!activeSession || !connected) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | null = null

    const poll = async () => {
      if (cancelled) return
      try {
        const res = await fetch(`/api/tasks/${activeSession}`)
        if (!res.ok || cancelled) return
        const data = await res.json()
        // a PIPELINE job (running or waiting its turn in the queue) is never a session load:
        // it has its own progress, and taking it for one hid the toolbar of a session whose
        // segmentation sat queued behind another scan for hours (USER 2026-10-09, observatorio)
        const tasks = ((data.tasks || []) as { label: string; pct: number; detail: string; task_type?: string }[])
          .filter(tk => tk.task_type !== 'pipeline')
        if (tasks && tasks.length > 0) {
          const t = tasks[0]  // show the first active task
          const msg = t.detail ? `${t.label} — ${t.detail}` : t.label
          // Don't set sessionLoading when pipeline is running — pipeline panel already shows progress
          const pr = pipelineRunningRef.current
          const pipelineActive = pr && (pr.status === 'running' || pr.status === 'queued') && pr.session_id === activeSession
          if (!pipelineActive) {
            taskOverlayRef.current = msg
            setSessionLoading(msg)
          }
          // Keep polling while tasks are active
          timer = setTimeout(poll, 3000)
        } else {
          // Tasks done — if overlay was showing a server task, dismiss it (the one THIS poll
          // put up: the closure's sessionLoading/pointCount are the values of the effect's start)
          if (taskOverlayRef.current) {
            const shown = taskOverlayRef.current
            taskOverlayRef.current = null
            setSessionLoading(cur => (cur === shown ? null : cur))
            // Refresh segments in case segmentation tasks finished
            try {
              await reloadSegments(activeSession)
              viewportRef.current?.refreshSegmentOBBs(activeSession)
            } catch { /* silent */ }
          }
        }
      } catch { /* server unreachable, stop polling */ }
    }

    // Check immediately on session load / reconnect
    const initialDelay = setTimeout(poll, 500)

    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
      clearTimeout(initialDelay)
    }
  }, [activeSession, connected])  

  // Poll frame-extraction progress until it finishes, then refresh the session
  // list so frameCount > 0 and the "+" button becomes the reconstruct hammer.
  const pollVideoExtraction = useCallback((sessionId: string) => {
    const tick = async () => {
      try {
        const r = await fetch(`/api/sessions/${sessionId}/video/extract_progress`)
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const p = await r.json()
        if (p.phase === 'extracting') {
          setExtractingSessions(prev => ({ ...prev, [sessionId]: p.pct || 0 }))
          setStatusMessage(tt('video.extracting', { saved: p.saved || 0, total: p.total ? `/${p.total}` : '', pct: p.pct || 0 }))
          setTimeout(tick, 1500)
        } else if (p.phase === 'error') {
          setExtractingSessions(prev => { const n = { ...prev }; delete n[sessionId]; return n })
          setStatusMessage(tt('video.extractFailed', { detail: p.error || '' }))
        } else {
          // done or idle  frames are on disk
          setExtractingSessions(prev => { const n = { ...prev }; delete n[sessionId]; return n })
          setStatusMessage(tt('video.extracted', { n: p.saved || p.frame_count || 0 }))
          connectToServer()
        }
      } catch {
        setExtractingSessions(prev => { const n = { ...prev }; delete n[sessionId]; return n })
        setStatusMessage(tt('video.lostConnection'))
      }
    }
    tick()
  }, [connectToServer])

  // Trigger the hidden file picker for a given session (the "+" button).
  const handlePickVideo = useCallback((sessionId: string) => {
    pendingVideoSession.current = sessionId
    if (videoInputRef.current) {
      videoInputRef.current.value = ''  // allow re-selecting the same file
      videoInputRef.current.click()
    }
  }, [])

  // Upload the chosen video and kick off background extraction.
  const handleVideoSelected = useCallback(async (file: File) => {
    const sessionId = pendingVideoSession.current
    pendingVideoSession.current = null
    if (!sessionId || !token) return
    setExtractingSessions(prev => ({ ...prev, [sessionId]: 0 }))
    setStatusMessage(tt('video.uploading', { name: file.name }))
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`/api/sessions/${sessionId}/video/upload`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        setExtractingSessions(prev => { const n = { ...prev }; delete n[sessionId]; return n })
        setStatusMessage(tt('video.uploadFailed', { detail: err.detail || `HTTP ${res.status}` }))
        return
      }
      pollVideoExtraction(sessionId)
    } catch (err: any) {
      setExtractingSessions(prev => { const n = { ...prev }; delete n[sessionId]; return n })
      setStatusMessage(tt('video.uploadFailed', { detail: err?.message ?? err }))
    }
  }, [token, pollVideoExtraction])

  const refreshFloorLevel = useCallback(async (sessionId: string) => {
    try {
      const res = await fetch(`/api/segmentation/floor_level/${sessionId}`)
      if (res.ok) {
        const d = await res.json()
        setFloorLevel({ candidates: d.candidates || [], selected: d.selected ?? null })
      }
    } catch { /* panel simply hides the combobox */ }
  }, [])

  // the EXPLICIT floor action (the floor combobox): the server applies it only under the
  // user's rule — docs/plan_determinismo.md point 161 — and never at load (point 158)
  const applyFloorLevel = useCallback(async (
    sessionId: string, mode: 'auto' | 'explicit', instanceId?: number
  ) => {
    try {
      const res = await fetch('/api/segmentation/level_floor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, mode, instance_id: instanceId }),
      })
      if (!res.ok) return
      const d = await res.json()
      if (d.candidates) setFloorLevel({ candidates: d.candidates, selected: d.selected ?? null })
      if (d.leveled && d.changed && d.matrix) {
        viewportRef.current?.setFloorTransform(d.matrix)
        viewportRef.current?.refreshSegmentOBBs(sessionId)
        setStatusMessage(tt('instances.floorLeveled', { mm: d.residual_before_mm ?? '' }))
      }
    } catch { /* non-fatal */ }
  }, [])

  // On session load (once the cloud is in the viewer) only the floor candidates are READ
  // for the combobox. OPENING NEVER WRITES (docs/plan_determinismo.md points 158 / 161,
  // 2026-10-08): the automatic re-levelling that ran here ('auto_if_needed' — a 1 cm /
  // 0.5° bar on an unseeded plane, rewriting the floor frame, every OBB and the store at
  // every open) is gone; levelling is the explicit floor action, judged by the user's rule.
  useEffect(() => {
    if (!activeSession || pointCount <= 0) return
    if (floorLevelCheckedRef.current === activeSession) return
    floorLevelCheckedRef.current = activeSession
    refreshFloorLevel(activeSession)
  }, [activeSession, pointCount, refreshFloorLevel])

  const handleReconstruct = useCallback(async (sessionId: string) => {
    setPipelineDialogSession(sessionId)
    setPipelineSegment([])
    // Fetch available scans for this project
    try {
      const res = await fetch(`/sessions/${sessionId}/scans`)
      if (res.ok) {
        const data = await res.json()
        const scans = data.scans || []
        setScansList(scans)
        // Pre-select only scans NOT currently being rebuilt
        const pr = pipelineRunningRef.current
        const isActive = pr && (pr.status === 'running' || pr.status === 'queued') && pr.session_id === sessionId
        // If pipeline is active but scans list is missing, treat ALL as rebuilding
        const rebuildingKeys = isActive ? (pr.scans || scans.map((s: any) => s.key)) : []
        const selected = scans.filter((s: any) => !rebuildingKeys.includes(s.key)).map((s: any) => s.key)
        setSelectedScans(selected)
        // Resume detection: if any selected scan has a partial (cached but
        // unfinished) reconstruction, default "Replace existing outputs" to OFF
        // so Run completes the missing parts instead of wiping the cache.
        const anyPartial = scans.some((s: any) => selected.includes(s.key) && s.recon_state === 'partial')
        setPipelineReplace(!anyPartial)
      } else {
        setScansList([])
        setSelectedScans([])
      }
    } catch {
      setScansList([])
      setSelectedScans([])
    }
    setPipelineDialogOpen(true)
  }, [])

  // "Reconstruir" goes over HTTP and NEVER loads the session (USER 2026-10-05: the
  // order used to travel on the viewer socket after loading the session, the socket
  // choked on the segmentation broadcast and the order was lost in silence — three
  // sent, one arrived). The jobs list polls /api/pipelines/active, so progress shows
  // whether or not a viewer socket is open.
  const handlePipelineRun = useCallback(async () => {
    if (!pipelineDialogSession) return
    if (selectedScans.length === 0) return
    const session = pipelineDialogSession
    const scans = [...selectedScans]
    setPipelineDialogOpen(false)
    try {
      const r = await fetch('/api/pipeline/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: session, replace: pipelineReplace, scans, segment: pipelineSegment.filter(k => scans.includes(k)) }),
      })
      const d = await r.json().catch(() => ({}))
      if (!r.ok || !d.ok) { setStatusMessage(d.detail || d.error || `${session}: reconstruction command rejected`); return }
      setPipelineRunning({ session_id: session, status: 'queued', current_stage_idx: -1, stages: [], scans })
      setStatusMessage(tt('pipeline.started', { session, n: scans.length }))
    } catch (e) {
      setStatusMessage(`${session}: ${e}`)
    }
  }, [pipelineDialogSession, pipelineReplace, selectedScans, pipelineSegment])

  // Autosegment (USER 2026-10-05): the segmentation chain on demand, prompts editable
  // USER 2026-11-05: "GenRecon es un botón más, como el de autosegmentar": the mesh of the OPEN scan, with the
  // confidence chosen in its window — an order of the GenRecon stage alone (pipeline.auto_genrecon is off)
  const [genreconOpen, setGenreconOpen] = useState(false)
  const runGenrecon = useCallback(async (confidence: number) => {
    if (!activeSession) return
    const rr = await fetch('/api/pipeline/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: activeSession, replace: false, scans: activeScanTab ? [activeScanTab] : [],
                             autosegment: { stages: { genrecon: true }, genrecon_confidence: confidence } }),
    })
    const dd = await rr.json().catch(() => ({}))
    if (!rr.ok || !dd.ok) { setStatusMessage(dd.detail || dd.error || 'GenRecon: command rejected'); return }
    setGenreconOpen(false)
    setStatusMessage(tt('genrecon.started', { session: activeSession, pct: Math.round(confidence * 100) }))
  }, [activeSession, activeScanTab])

  const openAutosegment = useCallback(async () => {
    if (!activeSession) return
    setAutosegState(null)
    setAutosegLoading(true)
    setAutosegOpen(true)
    // per SCAN (USER 2026-10-06): the open scan comes marked; the prompts shown are that scan's
    setAutosegScans(activeScanTab ? [activeScanTab] : [])
    try {
      const q = activeScanTab ? `?scan=${encodeURIComponent(activeScanTab)}` : ''
      const r = await fetch(`/api/autosegment/${activeSession}${q}`)
      const d = await r.json()
      if (d.ok) setAutosegState(d as AutosegmentState)
      else setStatusMessage(d.detail || d.error || 'autosegment: state unavailable')
    } catch (e) {
      setStatusMessage(`autosegment: ${e}`)
    } finally {
      setAutosegLoading(false)
    }
  }, [activeSession, activeScanTab])

  const runAutosegment = useCallback(async (vlmPrompt: string, sam3Prompts: string[], stages: AutosegmentStages) => {
    if (!activeSession || autosegScans.length === 0) return
    // the SAM3 prompts travel only when SAM3 runs WITHOUT a new VLM pass (a VLM pass makes them)
    const sam3List = stages.sam3 && !stages.vlm ? sam3Prompts : undefined
    // the prompts are saved in EACH marked scan (the session's record), then ONE order queues one task per scan.
    // A save the server refuses as BUSY (409: a job of that scan is queued or running with other prompts —
    // point 162) does not stop the order: the order carries its own prompts and the job writes them into the
    // scan when it runs (deliver_job_inputs); the refusal is only reported.
    const deferred: string[] = []
    for (const scan of autosegScans) {
      const body: Record<string, unknown> = { vlm_prompt: vlmPrompt, scan }
      if (sam3List) body.sam3_prompts = sam3List
      const r = await fetch(`/api/autosegment/${activeSession}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const d = await r.json().catch(() => ({}))
      if (!r.ok || !d.ok) {
        const busy = r.status === 409 && (d.busy === true || d.detail?.busy === true)
        if (!busy) { setStatusMessage(d.detail?.error || d.detail || d.error || `autosegment ${scan}: prompts not saved`); return }
        deferred.push(scan)
      }
    }
    // THE PROMPTS TRAVEL WITH THE ORDER (docs/plan_determinismo.md point 162): the command itself carries
    // the VLM prompt and the SAM3 list — the job runs with THESE, whatever the scan's files say by then
    const autosegment: Record<string, unknown> = { stages, vlm_prompt: vlmPrompt }
    if (sam3List) autosegment.sam3_prompts = sam3List
    const rr = await fetch('/api/pipeline/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: activeSession, replace: false, scans: autosegScans, autosegment }),
    })
    const dd = await rr.json().catch(() => ({}))
    if (!rr.ok || !dd.ok) { setStatusMessage(dd.detail || dd.error || 'autosegment: command rejected'); return }
    setAutosegOpen(false)
    const note = deferred.length ? ` — ${deferred.join(', ')}: a job is active, the queued job writes the prompts when it runs` : ''
    setStatusMessage(tt('autosegment.started', { session: `${activeSession} (${autosegScans.length})` }) + note)
  }, [activeSession, autosegScans])

  const handlePipelineCancel = useCallback(() => {
    const targetSession = pipelineRunning?.session_id || activeSession
    if (!targetSession) return
    viewportRef.current?.sendCommand({ type: 'cancel_pipeline', session_id: targetSession })
    setPipelineRunning(null)
    setStatusMessage(tt('pipeline.cancelled'))
  }, [activeSession, pipelineRunning])

  const handlePipelineProgress = useCallback((data: Record<string, unknown>) => {
    const newState: PipelineState = {
      session_id: (data.session_id as string) || undefined,
      status: (data.status as string) || 'running',
      current_stage_idx: (data.current_stage_idx as number) ?? -1,
      stages: (data.stages as PipelineStageInfo[]) || [],
    }
    const terminal = newState.status === 'done' || newState.status === 'failed'
      || newState.status === 'cancelled'
    // Exit the "stage pipeline" UI IMMEDIATELY on completion. The old code kept the
    // running state for 5s, so right after the cloud arrived the viewer still looked
    // like the pipeline was running ("vuelve a poner stage pipeline").
    if (terminal) {
      setPipelineRunning(null)
    } else {
      setPipelineRunning(prev => ({ ...newState, scans: prev?.scans }))
    }
    // Update status bar with current stage info (concise: stage + percentage only)
    const currentStage = newState.stages[newState.current_stage_idx]
    if (currentStage) {
      const pct = Math.round(currentStage.pct)
      // Short message for status bar; verbose detail stays in Pipeline dialog only
      const shortMsg = currentStage.message?.length > 60
        ? currentStage.message.slice(0, 57) + '…'
        : currentStage.message
      setStatusMessage(`${currentStage.label}: ${shortMsg || ''} ${pct} %`)
    }
    // Dead-socket FALLBACK only: if the WS reconnected during a long pipeline, the
    // backend's _on_pipeline_complete closure sent potree_ready on the now-dead socket,
    // so no cloud arrives on THIS live socket  reload the session. If the cloud DID load
    // here (the normal case), do NOTHING. The old code unconditionally clearScene()'d +
    // reloaded, which WIPED the cloud potree_ready had just loaded  the viewer got stuck
    // and only a backend restart + manual reload recovered it. No clearScene now: the
    // potree_ready handler already disposes/replaces the loader on reload.
    if (newState.status === 'done' && newState.session_id) {
      const sid = newState.session_id
      const doneAt = Date.now()
      setCertifyRefresh(v => v + 1)
      setTimeout(() => {
        if (lastCloudLoadAtRef.current >= doneAt) return  // cloud arrived on this socket  keep it
        if (activeSessionRef.current !== sid) return       // another session's job: the one on screen stays
        viewportRef.current?.sendCommandPreserveCamera({ type: 'load_session', session_id: sid })
      }, 8000)
    }
  }, [])

  const handleSegment = useCallback((sessionId: string) => {
    viewportRef.current?.sendCommand({ type: 'set_prompt', prompt: 'auto' })
    setStatusMessage(tt('segmentation.segmenting', { session: sessionId }))
  }, [])

  const handleUnload = useCallback(() => {
    viewportRef.current?.clearScene()
    viewportRef.current?.sendCommand({ type: 'cleared' })
    setActiveSession(null)
    setSegments([])
    setBimModels([])
    setPointCount(0)
    setCloudEverShown(false)
    setStatusMessage('')
  }, [])

  // ── Auto-load viewer prefs from server AFTER cloud finishes loading ──
  // Triggered when pointCount goes from 0 to >0 (cloud ready, all reset callbacks done)
  const prefsLoadedRef = useRef(false)
  const prevPointCountRef = useRef(0)
  useEffect(() => {
    const wasZero = prevPointCountRef.current === 0
    prevPointCountRef.current = pointCount
    if (!wasZero || pointCount === 0) return  // Only trigger on 0  N transition
    if (!activeSession || !token) return
    prefsLoadedRef.current = false
    fetch(`/api/sessions/${activeSession}/prefs`, {
      headers: { 'Authorization': `Bearer ${token}` }
    })
      .then(r => r.json())
      .then(prefs => {
        setPointSize(typeof prefs.pointSize === 'number' ? prefs.pointSize : 5.0)
        setConfidenceThreshold(typeof prefs.confidenceThreshold === 'number' ? prefs.confidenceThreshold : 0.0)
        setPointBudget(typeof prefs.pointBudget === 'number' ? prefs.pointBudget : 10_000_000)
        // USER 2026-11-05 ("la muestra instantáneamente y después la apaga"): a session OPENS with its cloud on screen.
        // Light 0 paints every point black (colour × light) and opacity 0 discards them — the state left behind
        // when the cloud was turned off to look at a mesh (pccr, zaragoza, station had both saved at 0). Those two
        // are restored only when they still show the cloud; 0 opens at 1.
        const visible = (v: unknown) => typeof v === 'number' && Number.isFinite(v) && v > 0
        setLightIntensity(visible(prefs.lightIntensity) ? prefs.lightIntensity : 1.0)
        setPointOpacity(visible(prefs.pointOpacity) ? prefs.pointOpacity : 1.0)
        setSceneLook(parseLook(prefs.sceneLook))
        setTimeout(() => { prefsLoadedRef.current = true }, 300)
      })
      .catch(() => {
        setPointSize(5.0)
        setConfidenceThreshold(0.0)
        setPointBudget(10_000_000)
        setLightIntensity(1.0)
        setPointOpacity(1.0)
        setSceneLook(DEFAULT_LOOK)
        prefsLoadedRef.current = true
      })
  }, [pointCount])  

  // ── Auto-save viewer prefs to server on change (debounced) ──
  const prefsSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (!prefsLoadedRef.current || !activeSession || !token) return
    if (prefsSaveTimer.current) clearTimeout(prefsSaveTimer.current)
    prefsSaveTimer.current = setTimeout(() => {
      fetch(`/api/sessions/${activeSession}/prefs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ pointSize, confidenceThreshold, pointBudget, lightIntensity, pointOpacity, sceneLook })
      }).catch(() => {})
    }, 500)
    return () => { if (prefsSaveTimer.current) clearTimeout(prefsSaveTimer.current) }
  }, [pointSize, confidenceThreshold, pointBudget, lightIntensity, pointOpacity, sceneLook])

  // ── Sábana: Generate comparison (auto_match  compare  show via Potree) ──
  // Open the comparison dialog (shows toggle inside)
  const handleGenerateComparison = useCallback(() => {
    if (!activeSession) return
    setCompareDialogOpen(true)
  }, [activeSession])

  // Actually run the comparison after dialog confirm
  const runComparison = useCallback(async () => {
    if (!activeSession) return
    setCompareDialogOpen(false)
    setSabanaLoading(true)
    setSabanaMetrics(null)
    setSabanaFullMeta(null)
    if (sabanaVisible) setSabanaVisible(false)
    viewportRef.current?.sendCommand({ type: 'load_session', session_id: activeSession })
    viewportRef.current?.setOBBsVisible(true)
    setActivePanel('bim')
    setStatusMessage(useManualAlignment ? tt('deviation.runningManual') : tt('deviation.runningAuto'))
    try {
      const matchRes = await fetch(`/api/bim/auto_match/${activeSession}`)
      const matchData = await matchRes.json()
      if (!matchData.matches?.length) {
        setStatusMessage(tt('deviation.noMatches'))
        setSabanaLoading(false)
        return
      }
      const compareRes = await fetch('/api/bim/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: activeSession,
          matches: matchData.matches,
          skip_registration: useManualAlignment,
        }),
      })
      const compareData = await compareRes.json()
      if (!compareData.ok) {
        setStatusMessage(tt('deviation.failed', { detail: compareData.error || '' }))
        setSabanaLoading(false)
        return
      }
      viewportRef.current?.sendCommand({ type: 'load_sabana', session_id: activeSession })
      viewportRef.current?.setOBBsVisible(false)
      if (activeTool === 'align') setActiveTool('navigate')
      setSabanaVisible(true)
      setSessions(prev => prev.map(s => s.id === activeSession ? { ...s, hasSabana: true } : s))
      try {
        const metaRes = await fetch(`/api/sessions/${activeSession}/sabana/meta`)
        if (metaRes.ok) {
          const fullMeta = await metaRes.json()
          setSabanaMetrics(fullMeta)
          setSabanaFullMeta(fullMeta)
          setActivePanel('analysis')
        }
      } catch { /* ignore */ }
    } catch (e: any) {
      setStatusMessage(tt('deviation.failed', { detail: e.message }))
    }
    setSabanaLoading(false)
  }, [activeSession, sabanaVisible, useManualAlignment])

  // ── Sábana: Toggle visibility via Potree streaming ──
  const handleToggleSabana = useCallback(() => {
    if (!activeSession) return
    if (sabanaVisible) {
      // Toggle OFF  reload original scan cloud + show OBBs (preserve camera!)
      viewportRef.current?.sendCommandPreserveCamera({ type: 'load_session', session_id: activeSession })
      viewportRef.current?.setOBBsVisible(true)
      setSabanaVisible(false)
      setSabanaMetrics(null)
      setSabanaFullMeta(null)
      setStatusMessage(tt('deviation.reloadingCloud'))
      setActivePanel('bim')
      return
    }
    // Toggle ON  load sábana + hide OBBs + fetch analysis meta
    setStatusMessage(tt('deviation.loading'))
    viewportRef.current?.sendCommand({ type: 'load_sabana', session_id: activeSession })
    viewportRef.current?.setOBBsVisible(false)
    if (activeTool === 'align') setActiveTool('navigate')
    setSabanaVisible(true)
    // Fetch full metadata  show analysis panel
    fetch(`/api/sessions/${activeSession}/sabana/meta`)
      .then(r => r.ok ? r.json() : null)
      .then(fullMeta => {
        if (fullMeta) {
          setSabanaMetrics(fullMeta)
          setSabanaFullMeta(fullMeta)
          setActivePanel('analysis')
        }
      })
      .catch(() => { })
  }, [activeSession, sabanaVisible])


  const hasSession = activeSession !== null
  // ── Shell wiring (prompt_ui.txt §5): layout regions, palette, toasts ──
  const { t, fmt, lang, setLang } = useI18n()
  const layout = useLayout()
  const { notify } = useToast()
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [colorMode, setColorMode] = useState<ColorMode>('rgb')
  const [mvThreshold, setMvThreshold] = useState(2)
  // hide the red band instead of painting it (kit "Votes"): the cloud
  // as it would look without its single-witness mass. Nothing is deleted.
  const [mvHide, setMvHide] = useState(false)
  const [selectedSegmentId, setSelectedSegmentId] = useState<number | null>(null)
  // Select an object and FLY TO it (USER 2026-09-21) — the instance list's click, and the
  // assistant's answer about an object (USER 2026-10-09: "debe mover la escena y enfocar
  // exactamente el elemento como si lo hubiera clickeado"). The frame comes from the
  // instance's own OBB, so a 9 m floor and a 20 cm light both fill the view; an instance with
  // no OBB (just propagated, no points) simply does not move the camera.
  const selectSegmentRef = useRef<(id: number | null) => void>(() => {})
  const selectSegment = useCallback((id: number | null) => selectSegmentRef.current(id), [])
  const focusSegmentsRef = useRef<(ids: number[]) => void>(() => {})
  const focusSegments = useCallback((ids: number[]) => focusSegmentsRef.current(ids), [])
  selectSegmentRef.current = (id: number | null) => {
    setSelectedSegmentId(id)
    if (id == null) return
    // the chat stays open when it is the inspector's tab (switching would close it)
    if (layout.state.inspectorTab !== 'assistant') layout.openInspector('properties')
    const seg = segments.find(s => s.id === id)
    if (seg?.focus) viewportRef.current?.flyToPoint(seg.focus.center, seg.focus.radius)
  }
  // several objects (a distance, a clearance): the first is selected, the view frames them all
  // (from the assistant: the chat stays open — no inspector switch, it would close the chat)
  focusSegmentsRef.current = (ids: number[]) => {
    const segs = ids.map(i => segments.find(s => s.id === i)).filter((s): s is SegmentInstance => !!s)
    if (!segs.length) return
    setSelectedSegmentId(segs[0].id)
    const framed = segs.filter(s => s.focus).map(s => s.focus!)
    if (!framed.length) return
    const c = [0, 1, 2].map(k => framed.reduce((a, f) => a + f.center[k], 0) / framed.length) as [number, number, number]
    const r = Math.max(...framed.map(f => Math.hypot(f.center[0] - c[0], f.center[1] - c[1], f.center[2] - c[2]) + f.radius))
    viewportRef.current?.flyToPoint(c, r)
  }
  // Evaluation volumes are the SCENE's, whoever made them (the assistant or the user): drawn,
  // listed in Instances, deletable — they do not leave with the chat (USER 2026-10-09)
  const [userVolumes, setUserVolumes] = useState<UserVolume[]>([])
  const refreshVolumes = useCallback(async () => {
    if (!activeSession) { setUserVolumes([]); return }
    try {
      const r = await fetch('/api/scene/volumes/list', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                        body: JSON.stringify({ session_id: activeSession }) })
      const d = await r.json()
      const vols: UserVolume[] = Array.isArray(d.volumes) ? d.volumes : []
      setUserVolumes(vols)
      for (const v of vols) viewportRef.current?.addUserVolume(v)
    } catch { /* no instance store yet */ }
  }, [activeSession])
  useEffect(() => { refreshVolumes() }, [refreshVolumes, cloudEverShown])
  const deleteVolume = useCallback(async (id: number) => {
    try {
      await fetch('/api/scene/volumes/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                 body: JSON.stringify({ session_id: activeSession, volume_id: id }) })
    } catch { /* the list below is refreshed either way */ }
    viewportRef.current?.removeUserVolume(id)
    setUserVolumes(prev => prev.filter(v => v.volume_id !== id))
  }, [activeSession])
  const focusVolume = useCallback((id: number) => {
    const v = userVolumes.find(x => x.volume_id === id)
    if (v) viewportRef.current?.flyToPoint(v.center as [number, number, number], Math.max(0.5, Math.hypot(v.size[0], v.size[1], v.size[2]) / 2))
  }, [userVolumes])
  // Esc deselects the object: its OBB and dimensions go (USER 2026-10-09)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      const el = e.target as HTMLElement | null
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return
      setSelectedSegmentId(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  // the list's OBB / visible boxes for every kind of row (USER 2026-10-09): OBBs off by default
  const [obbKeys, setObbKeys] = useState<Set<string>>(new Set())
  const [placedObb, setPlacedObb] = useState<Set<number>>(new Set())
  const [volumeHidden, setVolumeHidden] = useState<Set<number>>(new Set())
  // the cubes of the session's cut (USER 2026-11-06): listed in the Instances tab, each shown / hidden there;
  // another session starts with every cube hidden (the viewer forgets its choices with it)
  const [cubes, setCubes] = useState<CubeInfo[]>([])
  const [cubeVisible, setCubeVisible] = useState<Set<number>>(new Set())
  useEffect(() => { setCubes([]); setCubeVisible(new Set()) }, [activeSession])
  const [volumeObb, setVolumeObb] = useState<Set<number>>(new Set())
  const flip = <K,>(set: Set<K>, k: K, on: boolean) => { const n = new Set(set); if (on) n.add(k); else n.delete(k); return n }
  const toggleObb = useCallback((key: string, on: boolean) => {
    setObbKeys(prev => flip(prev, key, on)); viewportRef.current?.toggleOBB(key, on)
  }, [])
  const togglePlacedObb = useCallback((id: number, on: boolean) => {
    setPlacedObb(prev => flip(prev, id, on)); viewportRef.current?.setSceneObjectObb(id, on)
  }, [])
  const toggleVolume = useCallback((id: number, visible: boolean) => {
    setVolumeHidden(prev => flip(prev, id, !visible)); viewportRef.current?.setVolumeVisible(id, visible)
  }, [])
  const toggleVolumeObb = useCallback((id: number, on: boolean) => {
    setVolumeObb(prev => flip(prev, id, on)); viewportRef.current?.setVolumeObb(id, on)
  }, [])
  useEffect(() => { setObbKeys(new Set()); setPlacedObb(new Set()); setVolumeHidden(new Set()); setVolumeObb(new Set()) }, [activeSession])
  // THE CHAT DRIVES THE VIEWER (USER 2026-10-09): every `viewer` action its tools return is
  // performed here, with the same handlers the list and the view controls use
  const viewerActionRef = useRef<(a: Record<string, unknown>) => void>(() => {})
  const runViewerAction = useCallback((a: Record<string, unknown>) => viewerActionRef.current(a), [])
  viewerActionRef.current = (a: Record<string, unknown>) => {
    const ids = Array.isArray(a.ids) ? (a.ids as number[]) : []
    const vp = viewportRef.current
    const setVis = (pick: (s: SegmentInstance) => boolean | null) => {
      const next = segments.map(s => { const v = pick(s); return v == null ? s : { ...s, visible: v } })
      next.forEach(s => { vp?.setSegmentVisibility(s.classId, s.visible); vp?.toggleOBB(s.key, s.visible && obbKeys.has(s.key)) })
      setSegments(next)
    }
    switch (a.action) {
      case 'focus': focusSegmentsRef.current(ids); break
      case 'visibility': setVis(s => (ids.includes(s.id) ? !!a.visible : null)); break
      case 'show_only': setVis(s => ids.includes(s.id)); break
      case 'show_all': setVis(() => true); break
      case 'obb': ids.forEach(id => { const seg = segments.find(s => s.id === id); if (seg) toggleObb(seg.key, !!a.on) }); break
      case 'view': if (a.view === 'home') vp?.resetCamera(); else vp?.setStandardView(a.view as ViewPreset); break
      case 'orbit': {
        // a visible turn, not a jump: the orbit in steps over ~1 s
        const steps = 40, yaw = THREE_DEG * Number(a.yaw_deg || 0) / steps, pitch = -THREE_DEG * Number(a.pitch_deg || 0) / steps
        for (let k = 1; k <= steps; k++) setTimeout(() => viewportRef.current?.orbitBy(yaw, pitch), k * 25)
        break
      }
      case 'travel': {
        focusSegmentsRef.current(ids.slice(0, 1))
        if (ids.length > 1) setTimeout(() => focusSegmentsRef.current(ids.slice(1, 2)), 2200)
        break
      }
      case 'color': setColorMode(a.mode as ColorMode); break
      case 'clear': setSelectedSegmentId(null); vp?.clearAssistantViz(); break
    }
  }
  const [cursor, setCursor] = useState<{ x: number; y: number; z: number } | null>(null)
  const [readouts, setReadouts] = useState<ReadoutAnchor[]>([])
  // the scan's stored calibration measurements, drawn on the model: the ones of the CURRENT geometry epoch (a stale
  // one was picked on another cloud: its points are no longer on this one)
  const [calibrationSummary, setCalibrationSummary] = useState<CalSummary | null>(null)
  const drawCalibrationMarks = useCallback((measurements: CalMeasurement[], summary: CalSummary, planes: CalPlane[] = []) => {
    setCalibrationSummary(summary)
    const stale = new Set([...summary.rows.filter(r => r.stale).map(r => r.id), ...(summary.planes ?? []).filter(pl => pl.stale).map(pl => pl.id)])
    viewportRef.current?.setCalibrationMarks(
      measurements.filter(m => !stale.has(m.id)).map(m => ({ id: m.id, a: m.a, b: m.b, metres: m.measured_m, role: m.role })),
      planes.filter(pl => !stale.has(pl.id)).map(pl => ({ id: pl.id, points: pl.points, label: pl.label, inclinationDeg: pl.inclination_deg })))
  }, [])
  useEffect(() => {
    viewportRef.current?.setCalibrationMarks([])
    setCalibrationSummary(null)
    if (!activeSession || !cloudEverShown) return
    let cancelled = false
    fetch(`/api/calibration/${encodeURIComponent(activeSession)}${activeScanTab ? `?scan_key=${encodeURIComponent(activeScanTab)}` : ''}`)
      .then(r => (r.ok ? r.json() : null))
      .then(d => {
        if (cancelled || !d?.summary) return
        drawCalibrationMarks(d.document?.measurements ?? [], d.summary, d.document?.planes ?? [])
        // said as the cloud opens (USER 2026-11-13: "para mostrarle al usuario que esa nube aun no esta calibrada")
        if (d.summary.state === 'uncalibrated') setStatusMessage(tt('calibration.openUncalibrated'))
      })
      .catch(() => { /* a scan with no calibration yet */ })
    return () => { cancelled = true }
  }, [activeSession, activeScanTab, cloudEverShown, drawCalibrationMarks])
  const [metersPerPixel, setMetersPerPixel] = useState<number | null>(null)
  const [certifyState, setCertifyState] = useState<any>(null)
  const [hasWitness, setHasWitness] = useState(false)
  const [activeTasks, setActiveTasks] = useState<Array<{ label: string; pct: number; detail: string }>>([])
  const canManage = user?.role === 'admin' || user?.role === 'manager'

  // witness colour mode -> viewer (the certification kit shares this state)
  useEffect(() => {
    const mode = colorMode === 'status' || colorMode === 'mv_votes' ? colorMode : 'rgb'
    viewportRef.current?.setWitnessColorMode(mode, mvThreshold, mvHide && hasWitness)
  }, [colorMode, mvThreshold, mvHide, hasWitness, pointCount])
  useEffect(() => { if (colorMode === 'deviation' && !sabanaVisible) setColorMode('rgb') }, [sabanaVisible, colorMode])
  useEffect(() => { setColorMode('rgb'); setSelectedSegmentId(null); setReadouts([]) }, [activeSession])
  // geometry epoch for the status bar: certify state (re-read with the kit)
  useEffect(() => {
    if (!activeSession) { setCertifyState(null); return }
    let cancelled = false
    fetch(`/api/certify/state/${activeSession}`).then(r => (r.ok ? r.json() : null)).then(st => { if (!cancelled) setCertifyState(st) }).catch(() => {})
    return () => { cancelled = true }
  }, [activeSession, certifyRefresh])
  // the certification kit lives in the Inspector's Acta tab
  useEffect(() => { if (showCertifyKit) layout.openInspector('acta') }, [showCertifyKit])  
  // active server tasks for the Jobs dock
  useEffect(() => {
    if (!activeSession || !connected) { setActiveTasks([]); return }
    const iv = setInterval(async () => {
      try {
        const d = await fetch(`/api/tasks/${activeSession}`).then(r => r.json())
        setActiveTasks((d.tasks || []).map((x: any) => ({ label: x.label || x.task_type || '', pct: x.pct || 0, detail: x.detail || '' })))
      } catch { /* keep last */ }
    }, 4000)
    return () => clearInterval(iv)
  }, [activeSession, connected])

  // keyboard: palette, inspector, left panel, console, reset camera
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tgt = e.target as HTMLElement | null
      const typing = tgt && (tgt.tagName === 'INPUT' || tgt.tagName === 'TEXTAREA' || tgt.isContentEditable)
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setPaletteOpen(o => !o); return }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'j') { e.preventDefault(); layout.toggleInspector(); return }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') { e.preventDefault(); layout.setLeftTab(layout.state.leftTab ? null : 'sessions'); return }
      if ((e.ctrlKey || e.metaKey) && e.key === '`') { e.preventDefault(); if (layout.state.dockOpen && layout.state.dockTab === 'console') layout.setDockOpen(false); else layout.openDock('console'); return }
      if (typing) return
      if (e.key === 'Home') { viewportRef.current?.resetCamera(); e.preventDefault() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [layout])

  const report = useCallback((msg: string, tone: 'ok' | 'err' | 'warn' | 'info' = 'info') => {
    setStatusMessage(msg)
    if (tone !== 'info') notify(msg, { tone })
  }, [notify])

  const disconnect = useCallback((andLogout = false) => {
    viewportRef.current?.sendCommand({ type: 'cleared' })
    viewportRef.current?.clearScene()
    setConnected(false)
    setActivePanel('sessions')
    setSessions([])
    setActiveSession(null)
    setSelectedSession(null)
    setSessionLoading(null)
    setSegments([])
    setBimModels([])
    setPointCount(0)
    setCloudEverShown(false)
    setActiveTool('navigate')
    setStatusMessage('')
    if (andLogout) logout()
  }, [logout])  

  const openObjectLibrary = useCallback(async () => {
    setShowObjectLibrary(true)
    try {
      const r = await fetch('/api/objects/library')
      if (r.ok) setObjectLibrary((await r.json()).items || [])
    } catch { /* silent */ }
  }, [])

  const undoBrush = useCallback(async () => {
    if (!activeSession) return
    try {
      const r = await fetch('/api/segmentation/erase/undo', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: activeSession }) })
      const d = await r.json()
      report(d.ok ? t('brush.undone', { n: fmt.integer(d.restored || 0) }) : (d.detail || t('brush.nothingToUndo')), d.ok ? 'ok' : 'warn')
      if (d.ok) await reloadSegments(activeSession)
    } catch { report(t('brush.undoFailed'), 'err') }
  }, [activeSession, report, t, fmt, reloadSegments])

  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) document.exitFullscreen()
    else document.documentElement.requestFullscreen()
  }, [])

  const renameSession = useCallback(async (id: string, name: string) => {
    try {
      const headers: HeadersInit = { 'Content-Type': 'application/json' }
      if (token) headers['Authorization'] = `Bearer ${token}`
      const res = await fetch(`/sessions/${id}`, { method: 'PATCH', headers, body: JSON.stringify({ name }) })
      if (!res.ok) { const err = await res.json().catch(() => ({})); report(err.detail || t('sessions.renameFailed'), 'err') }
      else { report(t('sessions.renamed', { name }), 'ok'); connectToServer() }
    } catch { report(t('sessions.renameFailed'), 'err') }
  }, [token, connectToServer, report, t])

  const deleteSession = useCallback(async (id: string) => {
    const ok = await confirmDanger(t('sessions.deleteMessage', { id }), t('sessions.deleteTitle', { id }))
    if (!ok) return
    try {
      const headers: HeadersInit = {}
      if (token) headers['Authorization'] = `Bearer ${token}`
      const res = await fetch(`/sessions/${id}`, { method: 'DELETE', headers })
      if (!res.ok) { const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` })); report(err.detail || t('sessions.deleteFailed'), 'err') }
      else { if (activeSession === id) handleUnload(); report(t('sessions.deleted', { id }), 'ok'); connectToServer() }
    } catch (err: any) { report(t('sessions.deleteFailedDetail', { detail: err?.message || err }), 'err') }
  }, [token, activeSession, handleUnload, connectToServer, confirmDanger, report, t])

  const createSession = useCallback(async (name: string) => {
    try {
      const headers: HeadersInit = { 'Content-Type': 'application/json' }
      if (token) headers['Authorization'] = `Bearer ${token}`
      const res = await fetch('/sessions', { method: 'POST', headers, body: JSON.stringify({ name }) })
      if (!res.ok) { const err = await res.json().catch(() => ({})); report(err.detail || t('sessions.createFailed'), 'err') }
      else { report(t('sessions.created', { name }), 'ok'); connectToServer() }
    } catch { report(t('sessions.createFailed'), 'err') }
  }, [token, connectToServer, report, t])

  const setReference = useCallback(async (sid: string, sc: any) => {
    if (sc.is_reference || sc.kind === 'fused') return
    try {
      await fetch(`/api/project/${sid}/composition/reference`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scan_key: sc.key }) })
      report(t('scans.referenceSet', { label: sc.label }), 'ok')
      loadProjectScans(sid)
    } catch { report(t('scans.referenceFailed'), 'err') }
  }, [loadProjectScans, report, t])

  const renameSegment = useCallback(async (seg: SegmentInstance, newLabel: string) => {
    if (!activeSession) return
    try {
      const r = await fetch('/api/segmentation/rename', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: activeSession, instance_id: seg.id, label: newLabel }),
      })
      if (!r.ok) {
        const d = await r.json().catch(() => ({}))
        report(t('instances.renameFailed', { detail: d.detail || r.status }), 'err')
        return
      }
    } catch (e) { report(t('instances.renameFailed', { detail: String(e) }), 'err'); return }
    setSegments(prev => prev.map(s => (s.key === seg.key ? { ...s, label: newLabel } : s)))
    viewportRef.current?.refreshSegmentOBBs(activeSession)
  }, [activeSession, report, t])

  const deleteSegment = useCallback(async (seg: SegmentInstance) => {
    const ok = await confirmDanger(t('instances.deleteMessage', { label: seg.label }), t('instances.deleteTitle'))
    if (!ok || !activeSession) return
    const res = await fetch('/api/segmentation/delete', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      // label pins the EXACT row: instance_id alone collides across writers (strip the ' #N' suffix)
      body: JSON.stringify({ session_id: activeSession, instance_id: seg.id, label: seg.label.replace(/ #\d+$/, '') }),
    })
    if (res.ok) {
      setSegments(prev => prev.filter(s => s.key !== seg.key))
      viewportRef.current?.setSegmentVisibility(seg.classId, false)
      viewportRef.current?.refreshSegmentOBBs(activeSession)
      report(t('instances.deleted', { label: seg.label }), 'ok')
    } else report(t('instances.deleteFailed', { status: res.status }), 'err')
  }, [activeSession, confirmDanger, report, t])

  // a generated (ShapeR) object: rename = its label; delete = its WHOLE folder, after a confirmation
  // like a segment's (USER 2026-10-01)

  const deleteTsdfMesh = useCallback(async (m: MeshListItem) => {
    if (!activeSession) return
    const ok = await confirmDanger(t('instances.deleteMeshMessage', { label: m.label, folder: m.folder }), t('instances.deleteMeshTitle'))
    if (!ok) return
    try {
      const r = await fetch('/api/segmentation/tsdf/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: activeSession, folder: m.folder }) })
      const d = await r.json()
      if (d.ok) {
        report(t('instances.meshDeleted', { label: m.label }), 'ok')
        viewportRef.current?.clearTsdf()
        await refreshTsdfMeshList(activeSession)
        viewportRef.current?.reloadTsdf(activeSession)
      } else report(t('instances.meshDeleteFailed', { detail: d.detail || '' }), 'err')
    } catch { report(t('instances.meshDeleteFailed', { detail: '' }), 'err') }
  }, [activeSession, confirmDanger, refreshTsdfMeshList, report, t])

  const resumeRun = useCallback(async () => {
    if (!resumeDialog) return
    const sid = resumeDialog.session
    setResumeBusy(true)
    setResumeProgress(null)
    setStatusMessage(t('resume.statusResuming'))
    try {
      const r = await fetch('/api/segmentation/resume/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: sid }) })
      const d = await r.json().catch(() => ({}))
      report(r.ok ? t('resume.done') : t('resume.failed', { detail: d.detail || '' }), r.ok ? 'ok' : 'err')
    } catch {
      // the browser may drop a very long POST while the backend keeps working — wait for the task itself
      setStatusMessage(t('resume.waitingBackend'))
      for (let i = 0; i < 2400; i++) {
        await new Promise(r => setTimeout(r, 3000))
        const d = await fetch(`/api/tasks/${sid}`).then(r => r.json()).catch(() => null)
        if (d && !(d.tasks || []).some((x: any) => x.task_type === 'resume')) break
      }
      report(t('resume.finished'), 'ok')
    }
    setResumeBusy(false)
    setResumeDialog(null)
  }, [resumeDialog, report, t])

  const resumeCancel = useCallback(async () => {
    if (!resumeDialog) return
    const sid = resumeDialog.session
    setResumeBusy(true)
    setStatusMessage(t('resume.deleting'))
    try {
      const r = await fetch('/api/segmentation/resume/cancel', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: sid }) })
      const d = await r.json().catch(() => ({}))
      report(r.ok ? t('resume.deletedIncomplete', { n: d.deleted || 0 }) : t('resume.cancelFailed'), r.ok ? 'ok' : 'err')
    } catch { report(t('resume.cancelFailed'), 'err') }
    setResumeBusy(false)
    setResumeDialog(null)
  }, [resumeDialog, report, t])

  const closeSegmentationManager = useCallback(async (dirty: boolean) => {
    const sid = interactiveSessionId
    setInteractiveSessionId(null)
    if (!sid) return
    if (!dirty) {
      setStatusMessage(t('segmentation.closedNoChanges'))
      try {
        await reloadSegments(sid)
        viewportRef.current?.refreshSegmentOBBs(sid)
        viewportRef.current?.sendCommandPreserveCamera({ type: 'load_session', session_id: sid })
      } catch { /* silent */ }
      return
    }
    setSessionLoading(t('segmentation.refreshing'))
    setStatusMessage(t('segmentation.refreshingDbscan'))
    try {
      const res = await fetch('/api/segmentation/refresh', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: sid }) })
      if (res.ok) {
        const data = await res.json()
        if (data.queued) {
          // the re-projection (+ certification) is a JOB of the pipeline manager now
          // (docs/plan_determinismo.md point 160): the segments, the octree and the floor
          // arrive through the job's completion broadcast, like any other run; the floor is
          // re-levelled by the certification itself, never from here
          report('Re-projection queued — the segments and the octree arrive when the job ends', 'ok')
          setSessionLoading(null)
          return
        }
        report(t('segmentation.refreshed', { n: data.instances?.length || 0 }), 'ok')
        if (Array.isArray(data.instances)) {
          setSegments(data.instances.map((inst: any) => ({
            key: inst.global_id || `${inst.label}_${inst.instance_id || inst.id}`,
            id: inst.instance_id || inst.id,
            classId: (inst.class_byte ?? (inst.instance_id || inst.id)) as number,
            label: `${inst.label}`,
            color: inst.color || tokenColor('--measure'),
            totalPoints: inst.total_points || 0,
            visible: true,
            excluded: inst.excluded || false,
          })))
        }
        viewportRef.current?.refreshSegmentOBBs(sid)
        refreshUnsegmentedCount(sid)
        refreshFloorLevel(sid)
        viewportRef.current?.sendCommandPreserveCamera({ type: 'load_session', session_id: sid })
      }
    } catch { /* silent */ }
    finally { setSessionLoading(null) }
  }, [interactiveSessionId, reloadSegments, refreshUnsegmentedCount, refreshFloorLevel, report, t])

  // ── STAC's commands of the interface (the database of commands: server/commands; the
  //    executors: ui/src/commands/registry.ts). USER 2026-11-14: "le pedi que conecte con el
  //    servidor y no conecto" / "le pedi que abra pccr y no abrio nada" — every command the
  //    catalogue declares for the interface has its executor HERE (the type below does not
  //    compile otherwise). They read the app's CURRENT state through a ref: registered once.
  const cmd = useRef({
    connected, connectToServer, disconnect, sessions, activeSession, handleSessionLoad, handleUnload,
    projectScans, activeScanTab, activateScan, openFuse, layout, setLang, lang, pointSize, tsdfMeshes,
    openObjectLibrary, openAutosegment, handleReconstruct,
  })
  cmd.current = {
    connected, connectToServer, disconnect, sessions, activeSession, handleSessionLoad, handleUnload,
    projectScans, activeScanTab, activateScan, openFuse, layout, setLang, lang, pointSize, tsdfMeshes,
    openObjectLibrary, openAutosegment, handleReconstruct,
  }
  /** what the interface says of itself with every question to STAC (the model needs the
   *  sessions' names to open one) */
  const getUiState = useCallback(() => {
    const c = cmd.current
    return {
      connected: c.connected, session: c.activeSession, scan: c.activeScanTab,
      sessions: c.sessions.map(x => x.name),
      scans: c.projectScans.map((sc: any) => (sc.kind === 'fused' ? String(sc.label) : `${sc.date} ${sc.label}`)),
      fullscreen: !!document.fullscreenElement, lang: c.lang,
    }
  }, [])
  useEffect(() => {
    type AppCommandId = Exclude<UiCommandId, 'talk.set_user_name' | 'talk.forget_user_name'>
    const done: Outcome = { ok: true }
    const noSession: Outcome = { ok: false, reason: 'no_session' }
    const open = () => cmd.current.activeSession
    const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
    /** wait (bounded) for the app's state to follow an action: the ref is rewritten at each render */
    const until = async (test: () => boolean, ms: number) => {
      const end = Date.now() + ms
      while (!test() && Date.now() < end) await new Promise(r => setTimeout(r, 40))
      return test()
    }
    const ensureConnected = async () => {
      if (cmd.current.connected) return true
      await cmd.current.connectToServer()
      return until(() => cmd.current.connected, 4000)
    }
    const TOOL: Record<string, Tool> = {
      navigate: 'navigate', measure_distance: 'measure-distance', measure_angle: 'measure-angle',
      section_box: 'section-box', brush: 'erase', align: 'align',
    }
    const scanNames = (sc: any): string[] => {
      const date = String(sc.date || '')
      const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(date)
      // a scan is called by its date as people say it too: 2026-08-24 is "2408", "24/08"
      return [String(sc.key), date, String(sc.label || ''), `${date} ${sc.label || ''}`, ...(m ? [`${m[3]}${m[2]}`, `${m[3]}/${m[2]}`, `${m[3]} ${m[2]}`] : [])]
    }
    const executors: Executors<AppCommandId> = {
      'app.connect': async () => ((await ensureConnected()) ? done : { ok: false, reason: 'server_down' }),
      'app.disconnect': () => { cmd.current.disconnect(false); return done },
      'app.fullscreen': async ({ on }) => {
        const is = !!document.fullscreenElement
        const want = on ?? !is
        if (want !== is) {
          // the browser allows it only shortly after a click or a key (the microphone's press counts)
          try { if (want) await document.documentElement.requestFullscreen(); else await document.exitFullscreen() } catch { return { ok: false, reason: 'needs_gesture' } }
        }
        return { ok: true, state: want }
      },
      'app.open_settings': () => { setSettingsOpen(true); return done },
      'app.open_command_palette': () => { setPaletteOpen(true); return done },
      'app.set_language': ({ lang: l }) => { cmd.current.setLang(l); return done },
      'app.set_density': ({ density }) => { cmd.current.layout.setDensity(density); return done },
      'layout.show_panel': ({ panel }) => {
        if ((panel === 'instances' || panel === 'scans' || panel === 'bim') && !open()) return noSession
        cmd.current.layout.setLeftTab(panel); return done
      },
      'layout.hide_side_panel': () => { cmd.current.layout.setLeftTab(null); return done },
      'layout.show_inspector': ({ tab }) => { cmd.current.layout.openInspector(tab); return done },
      'layout.hide_inspector': () => { cmd.current.layout.setInspectorOpen(false); return done },
      'layout.show_dock': ({ tab }) => { cmd.current.layout.openDock(tab); return done },
      'layout.hide_dock': () => { cmd.current.layout.setDockOpen(false); return done },
      'sessions.open': async ({ session }) => {
        if (!(await ensureConnected())) return { ok: false, reason: 'server_down' }
        await until(() => cmd.current.sessions.length > 0, 2000)
        const found = findByName(cmd.current.sessions, session, x => [x.id, x.name])
        if (found.ambiguous) return { ok: false, reason: 'ambiguous', detail: found.ambiguous.map(x => x.name).join(', ') }
        if (!found.item) return { ok: false, reason: 'not_found', detail: session }
        cmd.current.handleSessionLoad(found.item.id)
        return { ok: true, detail: found.item.name }
      },
      'sessions.close': () => { if (!open()) return noSession; cmd.current.handleUnload(); return done },
      'sessions.list': async () => {
        if (!(await ensureConnected())) return { ok: false, reason: 'server_down' }
        await until(() => cmd.current.sessions.length > 0, 2000)
        const names = cmd.current.sessions.map(x => x.name)
        return names.length ? { ok: true, detail: names.join(', ') } : { ok: false, reason: 'nothing_to_do' }
      },
      'scans.open': ({ scan }) => {
        const sid = open()
        if (!sid) return noSession
        const found = findByName(cmd.current.projectScans as any[], scan, scanNames)
        if (found.ambiguous) return { ok: false, reason: 'ambiguous', detail: found.ambiguous.map((sc: any) => `${sc.date} ${sc.label}`).join(', ') }
        if (!found.item) return { ok: false, reason: 'not_found', detail: scan }
        const sc: any = found.item
        cmd.current.activateScan(sid, { key: sc.key, label: sc.label, date: sc.date, kind: (sc.kind || 'scan') as 'scan' | 'fused' })
        return { ok: true, detail: `${sc.date} ${sc.label}` }
      },
      'scans.list': () => {
        if (!open()) return noSession
        const names = (cmd.current.projectScans as any[]).map(sc => (sc.kind === 'fused' ? String(sc.label) : `${sc.date} ${sc.label}`))
        return names.length ? { ok: true, detail: names.join(', ') } : { ok: false, reason: 'nothing_to_do' }
      },
      'scans.open_fuse': () => {
        const sid = open()
        if (!sid) return noSession
        if ((cmd.current.projectScans as any[]).filter(sc => sc.kind !== 'fused').length < 2) return { ok: false, reason: 'nothing_to_do' }
        cmd.current.openFuse(sid); return done
      },
      'viewer.set_tool': ({ tool }) => { if (!open()) return noSession; setActiveTool(TOOL[tool]); return done },
      'viewer.clear_measurements': () => { if (!open()) return noSession; viewportRef.current?.clearMeasurements(); return done },
      'viewer.remove_section_box': () => { if (!open()) return noSession; viewportRef.current?.resetSectionBox(); setActiveTool('navigate'); return done },
      'viewer.show_unsegmented': ({ on }) => { if (!open()) return noSession; setUnsegmentedVisible(on); viewportRef.current?.setSegmentVisibility(0, on); return { ok: true, state: on } },
      'display.shadows': ({ on }) => { setShadowsOn(on); return { ok: true, state: on } },
      'display.axes': ({ on }) => { setShowAxes(on); return { ok: true, state: on } },
      'display.grid': ({ on }) => { setShowGrid(on); return { ok: true, state: on } },
      'display.camera_poses': ({ on }) => { if (!open()) return noSession; setShowCameraPoses(on); return { ok: true, state: on } },
      'display.points_as_cubes': ({ on }) => { setCubesOn(on); return { ok: true, state: on } },
      'display.point_size': ({ size, change }) => {
        if (!open()) return noSession
        // "larger" / "smaller": one step of the size, as a turn of the slider
        const now = cmd.current.pointSize
        const v = clamp(size ?? (change === 'larger' ? now * 1.5 : change === 'smaller' ? now / 1.5 : now), 0.01, 5)
        setPointSize(v)
        return { ok: true, detail: String(Math.round(v * 100) / 100) }
      },
      'display.point_opacity': ({ opacity }) => { if (!open()) return noSession; setPointOpacity(clamp(opacity, 0, 1)); return done },
      'display.light': ({ intensity }) => { if (!open()) return noSession; setLightIntensity(clamp(intensity, 0, 2)); return done },
      'display.scene_look': () => { if (!open()) return noSession; setSceneLookOpen(true); return done },
      'meshes.show': ({ on, mesh }) => {
        if (!open()) return noSession
        const all = cmd.current.tsdfMeshes
        if (!all.length) return { ok: false, reason: 'nothing_to_do' }
        let targets = all
        if (mesh) {
          const found = findByName(all, mesh, m => [m.folder, m.label])
          if (found.ambiguous) return { ok: false, reason: 'ambiguous', detail: found.ambiguous.map(m => m.label || m.folder).join(', ') }
          if (!found.item) return { ok: false, reason: 'not_found', detail: mesh }
          targets = [found.item]
        }
        const folders = new Set(targets.map(m => m.folder))
        setTsdfMeshes(prev => prev.map(x => (folders.has(x.folder) ? { ...x, visible: on } : x)))
        folders.forEach(f => viewportRef.current?.setTsdfVisibility(f, on))
        return { ok: true, state: on }
      },
      'meshes.wireframe': ({ on }) => {
        if (!open()) return noSession
        const all = cmd.current.tsdfMeshes
        if (!all.length) return { ok: false, reason: 'nothing_to_do' }
        setTsdfMeshes(prev => prev.map(x => ({ ...x, wireframe: on })))
        all.forEach(m => viewportRef.current?.setMeshWireframe('tsdf', m.folder, on))
        return { ok: true, state: on }
      },
      'meshes.open_genrecon': () => { if (!open()) return noSession; setGenreconOpen(true); return done },
      'calibration.measure_distance': () => {
        if (!open()) return noSession
        setCalibrating('distance'); setActiveTool('measure-distance'); setStatusMessage(tt('calibration.measureStatus')); return done
      },
      'calibration.measure_plane': () => {
        if (!open()) return noSession
        setCalibrating('plane'); setActiveTool('measure-plane'); setStatusMessage(tt('calibration.planeStatus')); return done
      },
      'calibration.open': () => { if (!open()) return noSession; setCalibrationDraft(null); setCalibrationOpen(true); return done },
      'calibration.open_reference': ({ mode }) => { if (!open()) return noSession; setReferenceDialog(mode); return done },
      'objects.open_library': () => { if (!open()) return noSession; void cmd.current.openObjectLibrary(); return done },
      'physics.panel': ({ on }) => { if (!open()) return noSession; setPhysicsOpen(on); return { ok: true, state: on } },
      'segmentation.open_autosegment': () => { if (!open()) return noSession; void cmd.current.openAutosegment(); return done },
      'pipeline.open': async ({ session }) => {
        let sid = open()
        if (session) {
          if (!(await ensureConnected())) return { ok: false, reason: 'server_down' }
          const found = findByName(cmd.current.sessions, session, x => [x.id, x.name])
          if (found.ambiguous) return { ok: false, reason: 'ambiguous', detail: found.ambiguous.map(x => x.name).join(', ') }
          if (!found.item) return { ok: false, reason: 'not_found', detail: session }
          sid = found.item.id
        }
        if (!sid) return noSession
        void cmd.current.handleReconstruct(sid)
        return done
      },
      'flythrough.open': () => { const sid = open(); if (!sid) return noSession; setFlythroughOpen(sid); return done },
    }
    return registerCommands(executors)
  // (every value is read through `cmd`; the setters are stable)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Menus (File / View / Tools / Help) — the same actions as before ──
  const menus: HeaderMenu[] = [
    { id: 'file', label: t('menu.file'), entries: [
      connected
        ? { id: 'disconnect', label: t('menu.disconnect'), icon: <ArrowUpFromLine aria-hidden />, onSelect: () => disconnect(false) }
        : { id: 'connect', label: t('menu.connect'), icon: <Plug aria-hidden />, onSelect: connectToServer },
      { type: 'separator', id: 's1' },
      { id: 'export', label: t('menu.exportCloud'), icon: <Upload aria-hidden />, shortcut: 'Mod+E', disabled: true, onSelect: () => {} },
      { type: 'separator', id: 's2' },
      { id: 'settings', label: t('menu.settings'), icon: <Settings aria-hidden />, shortcut: 'Mod+,', onSelect: () => setSettingsOpen(true) },
    ] },
    { id: 'view', label: t('menu.view'), entries: [
      { id: 'panel', label: layout.state.leftTab ? t('menu.hidePanel') : t('menu.showPanel'), icon: layout.state.leftTab ? <PanelLeftClose aria-hidden /> : <PanelLeftOpen aria-hidden />, shortcut: 'Mod+B', onSelect: () => layout.setLeftTab(layout.state.leftTab ? null : 'sessions') },
      { id: 'inspector', label: layout.state.inspectorOpen ? t('menu.hideInspector') : t('menu.showInspector'), icon: <Sparkles aria-hidden />, shortcut: 'Mod+J', onSelect: layout.toggleInspector },
      { type: 'separator', id: 's1' },
      { id: 'reset', label: t('menu.resetCamera'), icon: <Home aria-hidden />, shortcut: 'Home', disabled: !hasSession, onSelect: () => viewportRef.current?.resetCamera() },
      { id: 'fullscreen', label: t('toolbar.fullscreen'), icon: <Maximize aria-hidden />, shortcut: 'F11', onSelect: toggleFullscreen },
      { id: 'shadows', label: t('menu.shadows'), icon: <Sun aria-hidden />, checked: shadowsOn, onSelect: () => setShadowsOn(v => !v) },
      { id: 'cubes', label: t('menu.cubes'), icon: <Package aria-hidden />, checked: cubesOn, onSelect: () => setCubesOn(v => !v) },
      { id: 'flyers', label: t('menu.flyers'), icon: <Circle aria-hidden />, checked: flyersOn, disabled: !hasSession, onSelect: () => setFlyersOn(v => !v) },
      { id: 'monoDetail', label: t('menu.monoDetail'), icon: <Circle aria-hidden />, checked: monoDetailOn, disabled: !hasSession, onSelect: () => setMonoDetailOn(v => !v) },
      { id: 'mixedUnresolved', label: t('menu.mixedUnresolved'), icon: <Circle aria-hidden />, checked: unresolvedOn, disabled: !hasSession, onSelect: () => setUnresolvedOn(v => !v) },
      { type: 'separator', id: 's2' },
      { id: 'console', label: t('dock.console'), icon: <Monitor aria-hidden />, shortcut: 'Mod+`', checked: consoleOpen, onSelect: () => setConsoleOpen(!consoleOpen) },
      { id: 'jobs', label: t('dock.jobs'), icon: <ListTodo aria-hidden />, checked: layout.state.dockOpen && layout.state.dockTab === 'jobs', onSelect: () => layout.openDock('jobs') },
      { type: 'separator', id: 's3' },
      { id: 'axes', label: t('menu.axes'), icon: <Axis3D aria-hidden />, checked: showAxes, onSelect: () => setShowAxes(v => !v) },
      { id: 'grid', label: t('menu.grid'), icon: <Grid3X3 aria-hidden />, checked: showGrid, onSelect: () => setShowGrid(v => !v) },
      { id: 'sceneLook', label: t('menu.sceneLook'), icon: <Palette aria-hidden />, disabled: !hasSession, onSelect: () => setSceneLookOpen(true) },
      { id: 'density', label: t('settings.density'), icon: <SlidersHorizontal aria-hidden />, hint: t(`settings.${layout.state.density}`), onSelect: () => setSettingsOpen(true) },
    ] },
    { id: 'tools', label: t('menu.tools'), disabled: !hasSession, entries: [
      { id: 'navigate', label: t('toolbar.navigate'), icon: <RotateCcw aria-hidden />, shortcut: 'V', checked: activeTool === 'navigate', onSelect: () => setActiveTool('navigate') },
      { id: 'resetview', label: t('toolbar.resetView'), icon: <Home aria-hidden />, shortcut: 'Home', onSelect: () => viewportRef.current?.resetCamera() },
      { type: 'separator', id: 's1' },
      { id: 'distance', label: t('toolbar.measureDistance'), icon: <Ruler aria-hidden />, shortcut: 'M', checked: activeTool === 'measure-distance', onSelect: () => setActiveTool('measure-distance') },
      { id: 'angle', label: t('toolbar.measureAngle'), icon: <TriangleRight aria-hidden />, shortcut: 'A', checked: activeTool === 'measure-angle', onSelect: () => setActiveTool('measure-angle') },
      { id: 'clearm', label: t('toolbar.clearMeasurements'), icon: <Trash2 aria-hidden />, onSelect: () => viewportRef.current?.clearMeasurements() },
      { id: 'refdistance', label: t('toolbar.referenceDistance'), icon: <Pin aria-hidden />, disabled: !activeSession, onSelect: () => setReferenceDialog('distance') },
      { id: 'refplane', label: t('toolbar.referencePlane'), icon: <Pentagon aria-hidden />, disabled: !activeSession, onSelect: () => setReferenceDialog('plane') },
      { type: 'separator', id: 's2' },
      { id: 'brush', label: t('toolbar.brush'), icon: <Brush aria-hidden />, checked: activeTool === 'erase', onSelect: () => setActiveTool(activeTool === 'erase' ? 'navigate' : 'erase') },
      { id: 'undobrush', label: t('brush.undo'), icon: <Undo2 aria-hidden />, onSelect: undoBrush },
      { id: 'align', label: t('toolbar.alignCloud'), icon: <Move aria-hidden />, shortcut: 'G', disabled: sabanaVisible, checked: activeTool === 'align', onSelect: () => setActiveTool(activeTool === 'align' ? 'navigate' : 'align') },
      { id: 'addobj', label: t('toolbar.addObject'), icon: <Package aria-hidden />, onSelect: openObjectLibrary },
      { id: 'physics', label: t('physics.title'), icon: <Circle aria-hidden />, checked: physicsOpen, onSelect: () => setPhysicsOpen(v => !v) },
      { type: 'separator', id: 's3' },
      { id: 'section', label: t('toolbar.sectionBox'), icon: <Scissors aria-hidden />, shortcut: 'X', checked: activeTool === 'section-box', onSelect: () => setActiveTool('section-box') },
      { id: 'resetsection', label: t('toolbar.resetSection'), icon: <Unlock aria-hidden />, onSelect: () => { viewportRef.current?.resetSectionBox(); setActiveTool('navigate') } },
      { type: 'separator', id: 's4' },
      { id: 'segmentation', label: t('instances.segmentation'), icon: <Crosshair aria-hidden />, onSelect: () => activeSession && openSegmentationManager(activeSession) },
      { id: 'autosegment', label: t('instances.autosegment'), icon: <Wand2 aria-hidden />, disabled: !activeSession, onSelect: openAutosegment },
      { id: 'fuse', label: t('sessions.fuse'), icon: <Layers aria-hidden />, disabled: projectScans.filter(sc => sc.kind !== 'fused').length < 2, onSelect: () => activeSession && openFuse(activeSession) },
      ...(hasCameraPoses ? [{ id: 'poses', label: t('menu.cameraPoses'), icon: <Camera aria-hidden />, checked: showCameraPoses, onSelect: () => setShowCameraPoses(v => !v) } as MenuEntry] : []),
    ] },
    { id: 'help', label: t('menu.help'), entries: [
      { id: 'palette', label: t('palette.title'), icon: <Search aria-hidden />, shortcut: 'Mod+K', onSelect: () => setPaletteOpen(true) },
      { id: 'docs', label: t('menu.documentation'), icon: <BookOpen aria-hidden />, disabled: true, onSelect: () => {} },
      { id: 'shortcuts', label: t('menu.shortcuts'), icon: <Keyboard aria-hidden />, shortcut: 'Mod+/', disabled: true, onSelect: () => {} },
      { type: 'separator', id: 's1' },
      { id: 'about', label: t('menu.about'), icon: <Info aria-hidden />, disabled: true, onSelect: () => {} },
    ] },
  ]

  const userEntries: MenuEntry[] = [
    { type: 'header', id: 'who', label: `${user?.full_name || user?.username || ''} — ${t(`role.${user?.role || 'viewer'}`)}` },
    { type: 'separator', id: 's0' },
    ...(user?.role === 'admin' ? [{ id: 'admin', label: t('menu.userManagement'), icon: <Users aria-hidden />, onSelect: () => setAdminOpen(true) } as MenuEntry] : []),
    { id: 'settings', label: t('menu.settings'), icon: <Settings aria-hidden />, onSelect: () => setSettingsOpen(true) },
    { type: 'separator', id: 's1' },
    { id: 'logout', label: t('menu.logout'), icon: <LogOut aria-hidden />, onSelect: () => disconnect(true) },
  ]

  // ── Command palette: menus + navigation (§5) ──
  const commands: Command[] = [
    ...menus.flatMap(m => m.entries.filter((e): e is Extract<MenuEntry, { onSelect: () => void }> => !e.type || e.type === 'item')
      .map(e => ({ id: `${m.id}.${e.id}`, label: typeof e.label === 'string' ? e.label : e.id, group: m.label, icon: e.icon, shortcut: e.shortcut, disabled: e.disabled || m.disabled, run: e.onSelect }))),
    ...sessions.map(s => ({ id: `session:${s.id}`, label: s.name, group: t('sessions.title'), icon: <FolderOpen aria-hidden />, keywords: t('sessions.load'), disabled: !s.hasCloud, hint: s.id === activeSession ? t('sessions.loaded') : undefined, run: () => handleSessionLoad(s.id) })),
    ...segments.map(seg => ({ id: `segment:${seg.key}`, label: seg.label, group: t('instances.title'), icon: <Tag aria-hidden />, hint: fmt.integer(seg.totalPoints), run: () => { setSelectedSegmentId(seg.id); layout.setLeftTab('instances'); layout.openInspector('properties') } })),
    ...projectScans.map((sc: any) => ({ id: `scan:${sc.key}`, label: sc.kind === 'fused' ? sc.label : `${sc.date} ${sc.label}`, group: t('scans.title'), icon: <Layers aria-hidden />, run: () => activeSession && openScanFromTree(activeSession, sc) })),
    { id: 'nav.sessions', label: t('sessions.title'), group: t('palette.navigate'), icon: <FolderOpen aria-hidden />, run: () => layout.setLeftTab('sessions') },
    { id: 'nav.instances', label: t('instances.title'), group: t('palette.navigate'), icon: <Tag aria-hidden />, run: () => layout.setLeftTab('instances') },
    { id: 'nav.scans', label: t('scans.title'), group: t('palette.navigate'), icon: <Layers aria-hidden />, run: () => layout.setLeftTab('scans') },
    { id: 'nav.bim', label: t('bim.title'), group: t('palette.navigate'), icon: <Building2 aria-hidden />, run: () => layout.setLeftTab('bim') },
    { id: 'nav.team', label: t('team.title'), group: t('palette.navigate'), icon: <Users aria-hidden />, run: () => layout.setLeftTab('team') },
    { id: 'nav.assistant', label: t('assistant.title'), group: t('palette.navigate'), icon: <Sparkles aria-hidden />, run: () => layout.openInspector('assistant') },
    { id: 'nav.acta', label: t('inspector.acta'), group: t('palette.navigate'), icon: <FileCheck2 aria-hidden />, run: () => layout.openInspector('acta') },
    { id: 'nav.report', label: t('dock.report'), group: t('palette.navigate'), icon: <BarChart3 aria-hidden />, run: () => layout.openDock('report') },
    { id: 'nav.evaluation', label: t('dock.evaluation'), group: t('palette.navigate'), icon: <FlaskConical aria-hidden />, run: () => layout.openDock('evaluation') },
    { id: 'nav.timeline', label: t('dock.timeline'), group: t('palette.navigate'), icon: <Clock aria-hidden />, run: () => layout.openDock('timeline') },
  ]

  const activityItems: ActivityItem[] = [
    { id: 'sessions', icon: <FolderOpen aria-hidden />, label: t('sessions.title'), shortcut: 'Mod+B', badge: sessions.length || undefined, active: layout.state.leftTab === 'sessions', onClick: () => layout.toggleLeftTab('sessions') },
    { id: 'instances', icon: <Tag aria-hidden />, label: t('instances.title'), badge: segments.length || undefined, active: layout.state.leftTab === 'instances', disabled: !hasSession, onClick: () => layout.toggleLeftTab('instances') },
    { id: 'scans', icon: <Layers aria-hidden />, label: t('scans.title'), badge: projectScans.length || undefined, active: layout.state.leftTab === 'scans', disabled: !hasSession, onClick: () => layout.toggleLeftTab('scans') },
    { id: 'bim', icon: <Building2 aria-hidden />, label: t('bim.title'), badge: bimModels.length || undefined, active: layout.state.leftTab === 'bim', disabled: !hasSession, onClick: () => layout.toggleLeftTab('bim') },
  ]
  const activityBottom: ActivityItem[] = [
    { id: 'team', icon: <Users aria-hidden />, label: t('team.title'), active: layout.state.leftTab === 'team', onClick: () => layout.toggleLeftTab('team') },
    { id: 'console', icon: <Monitor aria-hidden />, label: t('dock.console'), shortcut: 'Mod+`', active: consoleOpen, onClick: () => setConsoleOpen(!consoleOpen) },
    { id: 'settings', icon: <Settings aria-hidden />, label: t('menu.settings'), onClick: () => setSettingsOpen(true) },
  ]

  const activeScanRow = projectScans.find((s: any) => s.key === activeScanTab)
  const pipelineActiveHere = !!(pipelineRunning && pipelineRunning.status === 'running' && pipelineRunning.session_id === activeSession)
  // USER 2026-11-05 (observatorio: "no muestra nada ni barra de herramientas ni nube" while the mask projection ran):
  // the cloud on disk is shown during any job that does not rebuild it — only a pending reconstruction / cloud stage
  // keeps the viewer from loading it
  const cloudRebuildingHere = pipelineActiveHere && (pipelineRunning?.stages || []).some(
    s => s.enabled && (s.id === 'reconstruction' || s.id === 'cloudcompy') && s.status !== 'done')
  // A cloud on screen is never covered: progress rides as a banner instead.
  const viewerHasCloud = pointCount > 0 || cloudEverShown
  // nor is a session that already PUBLISHED one (USER 2026-11-02: "ese panel debe ocultarse si ya hay épocas
  // publicadas, no puede quedar sobre la época"): a reload whose octree fetch was aborted mid-pipeline left
  // station's epoch 4 under the "Building the point cloud" splash for the whole GenRecon stage
  // epoch 0 is no proof of a cloud: after a Replace the session reports epoch 0 with output/ empty, and the
  // "!= null" test hid the progress panel for the whole reconstruction (USER 2026-11-08: "tiene que aparecer ahí")
  const sessionHasPublishedCloud = !!sessions.find(s => s.id === activeSession)?.hasCloud || (certifyState?.epoch ?? 0) > 0
  const loadingOverlay = sessionLoading && !pipelineActiveHere && !viewerHasCloud
  const pipelineStages: ProgressStage[] = (pipelineRunning?.stages || []).filter(s => s.enabled).map(s => ({ id: s.id, label: s.label, status: s.status, pct: s.pct, detail: s.message }))
  const currentStage = pipelineRunning?.stages[pipelineRunning.current_stage_idx]
  const epoch: number | null = certifyState?.epoch ?? null
  // how many OTHER epochs the session can be shown in — not a pending
  // verdict (USER 2026-09-16: nothing waits for approval any more)
  const storedEpochs: number = Math.max((certifyState?.epochs?.length ?? 0) - 1, 0)
  const selectedSegment = segments.find(s => s.id === selectedSegmentId) ?? null
  // the selected segment lights its OBB and shows H / W / D in the viewport (USER 2026-10-04)
  useEffect(() => { viewportRef.current?.setSelectedSegment(selectedSegment?.key ?? null) }, [selectedSegment?.key])
  const hasSabanaResult = !!sessions.find(s => s.id === activeSession)?.hasSabana

  // (a hook: before the early returns below — React counts hooks per render)
  // the Instances list's callbacks, stable (the list is memoized — see InstancesPanel)
  const instancesCb = useStableCallbacks({
    onFocusVolume: focusVolume,
    onToggleCube: (id: number, on: boolean) => {
                  setCubeVisible(prev => { const n = new Set(prev); if (on) n.add(id); else n.delete(id); return n })
                  viewportRef.current?.setCubeVisible(id, on)
                },
    onSetAllCubes: (on: boolean) => {
                  setCubeVisible(on ? new Set(cubes.map(c => c.id)) : new Set())
                  cubes.forEach(c => viewportRef.current?.setCubeVisible(c.id, on))
                },
    onFocusCube: (id: number) => viewportRef.current?.focusCube(id),
    onDeleteVolume: deleteVolume,
    onToggleObb: toggleObb,
    onTogglePlacedObb: togglePlacedObb,
    onToggleVolume: toggleVolume,
    onToggleVolumeObb: toggleVolumeObb,
    onSelectSegment: selectSegment,
    onSetAllVisible: (v: boolean, listed: ListedRows) => {
                  // every row the list shows (USER 2026-10-01): segments, Unsegmented, placed objects,
                  // generated objects, meshes
                  const keys = new Set(listed.segmentKeys)
                  setSegments(prev => prev.map(s => (keys.has(s.key) ? { ...s, visible: v } : s)))
                  segments.forEach(s => { if (keys.has(s.key)) { viewportRef.current?.toggleOBB(s.key, v && obbKeys.has(s.key)); viewportRef.current?.setSegmentVisibility(s.classId, v) } })
                  if (listed.unsegmented) { setUnsegmentedVisible(v); viewportRef.current?.setSegmentVisibility(0, v) }
                  listed.placedIds.forEach(id => viewportRef.current?.setSceneObjectVisible(id, v))
                  const tf = new Set(listed.tsdfFolders)
                  setTsdfMeshes(prev => prev.map(x => (tf.has(x.folder) ? { ...x, visible: v } : x)))
                  tsdfMeshes.forEach(m => { if (tf.has(m.folder)) viewportRef.current?.setTsdfVisibility(m.folder, v) })
                },
    onToggleSegment: (seg, vis) => { setSegments(prev => prev.map(s => (s.key === seg.key ? { ...s, visible: vis } : s))); viewportRef.current?.toggleOBB(seg.key, vis && obbKeys.has(seg.key)); viewportRef.current?.setSegmentVisibility(seg.classId, vis) },
    onRenameSegment: renameSegment,
    onDeleteSegment: deleteSegment,
    onToggleUnsegmented: v => { setUnsegmentedVisible(v); viewportRef.current?.setSegmentVisibility(0, v) },
    onFloorLevel: (iid: number) => { if (activeSession) applyFloorLevel(activeSession, 'explicit', iid) },
    onTogglePlaced: (id, v) => viewportRef.current?.setSceneObjectVisible(id, v),
    onRemovePlaced: id => viewportRef.current?.removeSceneObject(id),
    onToggleTsdf: (m, v) => { setTsdfMeshes(prev => prev.map(x => (x.folder === m.folder ? { ...x, visible: v } : x))); viewportRef.current?.setTsdfVisibility(m.folder, v) },
    onToggleTsdfWireframe: (m, on) => { setTsdfMeshes(prev => prev.map(x => (x.folder === m.folder ? { ...x, wireframe: on } : x))); viewportRef.current?.setMeshWireframe('tsdf', m.folder, on) },
    onDeleteTsdf: deleteTsdfMesh,
    onRenamePlaced: async (id, name) => {
                  if (!activeSession) return
                  try {
                    const r = await fetch('/api/objects/scene/update', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: activeSession, id, name }) })
                    if (r.ok) setPlacedObjects(prev => prev.map(o => (o.id === id ? { ...o, name } : o)))
                    else report(t('instances.renameFailed', { status: r.status }), 'err')
                  } catch { report(t('instances.renameFailed', { status: '' }), 'err') }
                },
    onFocusItem: (kind, key, instanceId) => {
                  // the object's own bounds; a mesh not loaded yet → its instance's OBB
                  if (viewportRef.current?.flyToObject(kind, key)) return
                  const seg = instanceId != null ? segments.find(s => s.id === instanceId) : undefined
                  if (seg?.focus) viewportRef.current?.flyToPoint(seg.focus.center, seg.focus.radius)
                },
    onOpenSegmentation: () => { if (activeSession) openSegmentationManager(activeSession) },
    onOpenAutosegment: openAutosegment,
    onOpenGenrecon: () => { if (activeSession) setGenreconOpen(true) },
  })

  if (authLoading) {
    return (
      <div className="stac-login">
        <SplashOverlay title={t('app.name')} status={t('common.loading')} boot />
      </div>
    )
  }

  if (!user) return <LoginPage />


  return (
    <div
      className={`stac-app ${layout.state.leftTab ? '' : 'stac-app--left-closed'} ${layout.state.inspectorOpen ? '' : 'stac-app--inspector-closed'}`.trim()}
      style={{ '--left-w': `${layout.state.leftWidth}px`, '--inspector-w': `${layout.state.inspectorWidth}px`, '--dock-h': `${layout.state.dockHeight}px` } as React.CSSProperties}
    >
      <input ref={videoInputRef} type="file" className="stac-hidden"
        accept="video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,.mp4,.avi,.mov,.mkv,.m4v"
        onChange={e => { const f = e.target.files?.[0]; if (f) handleVideoSelected(f) }} />

      <div className="stac-app__header">
        <AppHeader menus={menus} connected={connected}
          sessions={sessions.map(s => ({ id: s.id, name: s.name, loaded: s.id === activeSession, hasCloud: s.hasCloud, pipeline: pipelineJobs[s.id] }))}
          activeSession={activeSession} onSelectSession={handleSessionLoad}
          scans={projectScans.map((s: any) => ({ key: s.key, label: s.label, date: s.date, kind: s.kind || 'scan', isReference: !!s.is_reference }))}
          activeScan={activeScanTab} onSelectScan={key => { const sc = projectScans.find((s: any) => s.key === key); if (sc && activeSession) activateScan(activeSession, { key: sc.key, label: sc.label, date: sc.date, kind: sc.kind || 'scan' }) }}
          onOpenPalette={() => setPaletteOpen(true)} inspectorOpen={layout.state.inspectorOpen} onToggleInspector={layout.toggleInspector}
          vlmStatus={vlmStatus} userName={user.username} userEntries={userEntries} />
      </div>

      <div className="stac-app__activity">
        <ActivityBar items={activityItems} bottom={activityBottom} ariaLabel={t('layout.activityBar')} />
      </div>

      <div className="stac-app__left">
        {layout.state.leftTab && (
          <SidePanel>
            {layout.state.leftTab === 'sessions' && (
              <SessionsPanel connected={connected} sessions={sessions} selectedSession={selectedSession} activeSession={activeSession}
                scansOf={sid => (sid === activeSession && projectScans.length ? projectScans : (allProjectScans[sid] || []))}
                activeScanKey={activeScanTab} pipelineJobs={pipelineJobs}
                extracting={extractingSessions} canManage={canManage}
                onConnect={connectToServer} onSelect={handleSessionSelect} onLoad={handleSessionLoad}
                onFlythrough={id => { if (activeSession !== id) handleSessionLoad(id); setFlythroughOpen(id) }}
                onReconstruct={handleReconstruct} onPickVideo={handlePickVideo} onSegment={handleSegment} onOpenManager={openSegmentationManager}
                onUnload={handleUnload} onRename={renameSession} onDelete={deleteSession} onCreate={createSession}
                onFuse={openFuse} onOpenScan={openScanFromTree} onSetReference={setReference} />
            )}
            {layout.state.leftTab === 'instances' && activeSession && (
              <InstancesPanel {...instancesCb} cubes={cubes} cubeVisible={cubeVisible} volumes={userVolumes}   obbKeys={obbKeys}  placedObb={placedObb}  volumeHidden={volumeHidden} volumeObb={volumeObb}   segments={segments} unsegmentedVisible={unsegmentedVisible} unsegmentedCount={unsegmentedCount} absorbedCount={absorbedCount} floorLevel={floorLevel}
                placedObjects={placedObjects} tsdfMeshes={tsdfMeshes} selectedSegmentId={selectedSegmentId}
 />
            )}
            {layout.state.leftTab === 'scans' && (
              <ScansPanel sessionId={activeSession} scans={projectScans} activeScanKey={activeScanTab}
                onOpenScan={sc => activeSession && openScanFromTree(activeSession, sc)} onSetReference={sc => activeSession && setReference(activeSession, sc)} />
            )}
            {layout.state.leftTab === 'bim' && (
              <BIMNavigator models={bimModels} userRole={activeSession ? (user?.role || 'viewer') : 'viewer'}
                onToggleVisibility={(names, v) => viewportRef.current?.toggleBIMVisibility(names, v)}
                onSelectElement={names => viewportRef.current?.highlightBIMElement(names)}
                onSetOpacity={(names, o) => viewportRef.current?.setBIMOpacity(names, o)}
                onUploadIFC={async file => {
                  if (!activeSession || !token) return
                  const formData = new FormData(); formData.append('file', file)
                  try {
                    const res = await fetch(`/api/sessions/${activeSession}/bim/upload`, { method: 'POST', headers: { 'Authorization': `Bearer ${token}` }, body: formData })
                    if (res.ok) {
                      setStatusMessage(t('bim.uploadedLoading', { name: file.name }))
                      const { loadIFC } = await import('./components/IFCLoader')
                      const result = await loadIFC(`/api/sessions/${activeSession}/bim/${file.name}`, file.name)
                      viewportRef.current?.addBIMGroup(result.group)
                      setBimModels(prev => [...prev, result])
                      report(t('bim.loaded', { name: file.name, n: result.group.children.length }), 'ok')
                      connectToServer()
                    } else report(t('bim.uploadFailed', { detail: (await res.json()).detail }), 'err')
                  } catch (err: any) { report(t('bim.uploadFailed', { detail: err.message }), 'err') }
                }}
                onDeleteIFC={async filename => {
                  if (!activeSession || !token) return
                  const ok = await confirmDanger(t('bim.deleteMessage', { name: filename }), t('bim.deleteTitle'))
                  if (!ok) return
                  try {
                    const res = await fetch(`/api/sessions/${activeSession}/bim/${filename}`, { method: 'DELETE', headers: { 'Authorization': `Bearer ${token}` } })
                    if (res.ok) { report(t('bim.deleted', { name: filename }), 'ok'); viewportRef.current?.removeBIMGroup(filename); setBimModels(prev => prev.filter(m => m.filename !== filename)); connectToServer() }
                    else report(t('bim.deleteFailed', { detail: (await res.json()).detail }), 'err')
                  } catch (err: any) { report(t('bim.deleteFailed', { detail: err.message }), 'err') }
                }} />
            )}
            {layout.state.leftTab === 'team' && <TeamPanel onCallUser={(userId, username) => setCallTarget({ userId, username })} />}
          </SidePanel>
        )}
      </div>

      <main className="stac-app__main">
        {activeSession && scanTabs.length > 0 && (
          <Tabs<string> className="stac-main__tabs" variant="chrome" size="sm" ariaLabel={t('scans.openTabs')}
            items={scanTabs.map(tab => {
              const sc = projectScans.find((s: any) => s.key === tab.key)
              return {
                id: tab.key, closable: true, title: `${tab.date} ${tab.label}${sc?.is_reference ? ` (${t('header.reference')})` : ''}`,
                icon: sc?.is_reference ? <Star className="stac-scantabs__ref" aria-hidden /> : tab.kind === 'fused' ? <Layers aria-hidden /> : undefined,
                label: tab.kind === 'fused' ? tab.label : `${tab.date} ${tab.label}`,
              }
            })}
            value={activeScanTab} onChange={key => { const tab = scanTabs.find(x => x.key === key); if (tab && tab.key !== activeScanTab) activateScan(activeSession, tab) }} onClose={closeScanTab} />
        )}

        <div className={`stac-main__viewport ${loadingOverlay ? 'stac-main__viewport--hidden' : ''} ${flythroughOpen ? 'stac-main__viewport--flythrough' : ''}`.trim()}>
          <Viewport
            ref={viewportRef}
            openScanKey={openScanKey}
            pointSize={pointSize}
            pointBudget={pointBudget}
            confidenceThreshold={sabanaVisible ? 0.0 : confidenceThreshold}
            activeSession={activeSession}
            activeTool={activeTool}
            eraseRadius={eraseRadius}
            eraseShape={eraseShape}
            eraseYawDeg={eraseYawDeg}
            onEraseRadiusChange={setEraseRadius}
            onEraseMarksChanged={setEraseMarks}
            onEraseBoxSelected={setEraseBoxSel}
            showAxes={showAxes}
            showGrid={showGrid}
            sceneLook={sceneLook}
            pipelineRunning={cloudRebuildingHere}
            onPointCount={n => { setPointCount(n); if (n > 0) { lastCloudLoadAtRef.current = Date.now(); setCloudEverShown(true) } }}
            onFps={setFps}
            onStatusMessage={setStatusMessage}
            onSegments={list => { setSegments(list); refreshUnsegmentedCount(activeSession) }}
            onEraseLedger={() => reloadSegments(activeSession)}
            onTsdfReady={(sid, stage) => { refreshTsdfMeshList(sid, stage === 'session-open') }}
            onSceneObjectsChanged={setPlacedObjects}
            onSceneObjectSelected={id => { setSceneObjSel(id); setAlignTarget(''); setAlignTargets(id != null ? (viewportRef.current?.getSceneAlignTargets() || []) : []) }}
            onPipelineProgress={handlePipelineProgress}
            onVolumeChanged={async params => {
              if (!activeSession) return
              try {
                await fetch('/api/scene/volumes/update', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: activeSession, ...params }) })
                const er = await fetch('/api/scene/volumes/evaluate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: activeSession, volume_id: params.volume_id }) })
                const ed = await er.json()
                const occ = 1 - (typeof ed.free_fraction === 'number' ? ed.free_fraction : 1)
                viewportRef.current?.setVolumeStatus(params.volume_id, occ < 0.02 ? 'free' : occ < 0.12 ? 'touching' : 'colliding')
              } catch { /* non-fatal: the volume keeps its last colour */ }
            }}
            onVolumeDeleted={async volumeId => {
              if (!activeSession) return
              try { await fetch('/api/scene/volumes/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: activeSession, volume_id: volumeId }) }) } catch { /* ignore */ }
            }}
            onHasConfidence={setHasConfidence}
            onHasWitness={setHasWitness}
            onBimLoaded={models => { setBimModels(models); if (models.length > 0) { setShowAxes(false); setShowGrid(false) } }}
            onSabanaLoaded={nPts => { setSabanaLoading(false); setSessionLoading(null); setStatusMessage(t('deviation.loadedPoints', { n: fmt.integer(nPts ?? 0) })) }}
            showCameraPoses={showCameraPoses && !sabanaVisible}
            onCubes={setCubes}
            onHasCameraPoses={setHasCameraPoses}
            onCursor={setCursor}
            onReadouts={setReadouts}
            onDistanceMeasured={m => {
              // armed from the Calibration button: the distance just measured IS the calibration's draft
              if (calibratingRef.current !== 'distance') return
              setCalibrationDraft({ kind: 'distance', readoutId: m.id, a: m.a, b: m.b, metres: m.metres })
              setCalibrationOpen(true); setCalibrating(null); setActiveTool('navigate')
            }}
            onPlaneMeasured={pl => {
              // a plane is only ever marked for a calibration: the window opens with it as its draft
              setCalibrationDraft({ kind: 'plane', readoutId: pl.id, points: pl.points })
              setCalibrationOpen(true); setCalibrating(null); setActiveTool('navigate')
            }}
            onSegmentPicked={cls => {
              // a point of an object: select it (OBB + dimensions) and travel to it; a click on
              // nothing or on unsegmented points deselects (USER 2026-10-09)
              const seg = segments.find(s => s.classId === cls)
              if (seg) { layout.setLeftTab('instances'); selectSegment(seg.id) } else selectSegment(null)
            }}
            onViewScale={setMetersPerPixel}
          />

          {hasSession && !sessionLoading && (
            <FloatingToolbar
              activeTool={activeTool} onTool={setActiveTool} onResetCamera={() => viewportRef.current?.resetCamera()}
              onClearMeasurements={() => viewportRef.current?.clearMeasurements()} onAddObject={openObjectLibrary}
              calibrating={calibrating}
              onCalibration={kind => {
                if (kind === 'list') { setCalibrationDraft(null); setCalibrationOpen(true); return }
                setCalibrating(kind)
                setActiveTool(kind === 'distance' ? 'measure-distance' : 'measure-plane')
                setStatusMessage(tt(kind === 'distance' ? 'calibration.measureStatus' : 'calibration.planeStatus'))
              }}
              onResetSection={() => viewportRef.current?.resetSectionBox()} sabanaVisible={sabanaVisible}
              brush={{
                shape: eraseShape, onShape: setEraseShape, radiusM: eraseRadius, onRadiusM: setEraseRadius, yawDeg: eraseYawDeg, onYawDeg: setEraseYawDeg,
                hasConfidence, confThr: eraseConfThr, confArmed: eraseConfArmed,
                onConfThr: v => { setEraseConfThr(v); setEraseConfArmed(true); viewportRef.current?.setConfHighlight(v) },
                onApplyConf: async () => { await viewportRef.current?.applyConfidenceFilter(eraseConfThr, segments.filter(s => s.visible).map(s => s.id), unsegmentedVisible); setEraseConfArmed(false) },
                onCancelConf: () => { setEraseConfArmed(false); viewportRef.current?.setConfHighlight(null) },
                boxSelected: eraseBoxSel, onBoxMode: m => viewportRef.current?.setEraseBoxMode(m), onRemoveBox: () => viewportRef.current?.removeSelectedEraseBox(),
                marks: eraseMarks, onErase: () => viewportRef.current?.commitErase(null, undefined, segments.filter(s => s.visible).map(s => s.id)),
                targets: segments.map(s => ({ id: s.id, label: s.label })), target: eraseTarget, onTarget: setEraseTarget,
                onAssign: () => {
                  const visibles = segments.filter(s => s.visible).map(s => s.id)
                  if (eraseTarget === 'new') {
                    const name = window.prompt(t('brush.newSegmentPrompt'))
                    if (!name || !name.trim()) return
                    viewportRef.current?.commitErase(null, name.trim(), visibles, unsegmentedVisible)
                  } else viewportRef.current?.commitErase(Number(eraseTarget), undefined, visibles, unsegmentedVisible)
                  setEraseTarget('')
                },
                onClearMarks: () => viewportRef.current?.clearEraseMarks(), onUndo: undoBrush,
                onMerge: async (target, sources) => {
                  if (!activeSession) return
                  try {
                    const r = await fetch('/api/segmentation/merge', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: activeSession, target_id: target, source_ids: sources }) })
                    const d = await r.json()
                    if (!r.ok) { report(t('brush.mergeFailed', { detail: d.detail || r.status }), 'err'); return }
                    const tgt = segments.find(s => s.id === target)
                    report(t('brush.merged', { n: sources.length, target: tgt ? `${tgt.label} ${tgt.id}` : String(target) }), 'ok')
                    await reloadSegments(activeSession)
                    viewportRef.current?.refreshSegmentOBBs(activeSession)
                  } catch (e) { report(t('brush.mergeFailed', { detail: String(e) }), 'err') }
                },
              }}
              display={{ pointSize, onPointSize: setPointSize, light: lightIntensity, onLight: setLightIntensity, pointOpacity, onPointOpacity: setPointOpacity, budgetM: pointBudget / 1_000_000, onBudgetM: v => setPointBudget(v * 1_000_000), hasConfidence, confidence: confidenceThreshold, onConfidence: setConfidenceThreshold }}
              view={{ hasCameraPoses, showCameraPoses, onToggleCameraPoses: () => setShowCameraPoses(v => !v), showGrid, onToggleGrid: () => setShowGrid(v => !v), showAxes, onToggleAxes: () => setShowAxes(v => !v), onFullscreen: toggleFullscreen }}
              color={{
                mode: sabanaVisible ? 'deviation' : colorMode, hasWitness, hasDeviation: hasSabanaResult, mvThreshold, onMvThreshold: setMvThreshold,
                onMode: m => { if (m === 'deviation') { if (!sabanaVisible) handleToggleSabana() } else { if (sabanaVisible) handleToggleSabana(); setColorMode(m) } },
              }}
              compare={{
                available: bimModels.length > 0 && segments.length > 0, loading: sabanaLoading, hasResult: hasSabanaResult, showing: sabanaVisible,
                onRun: handleGenerateComparison, onToggle: handleToggleSabana,
                chip: sabanaMetrics ? t('deviation.elementsChip', { n: sabanaMetrics.elements?.filter((e: any) => e.status === 'evaluated').length || 0, total: sabanaMetrics.summary?.total_elements || 0 }) : undefined,
              }}
              placed={sceneObjSel != null ? {
                name: placedObjects.find(o => o.id === sceneObjSel)?.name || t('placed.fallbackName', { id: sceneObjSel }),
                onMode: m => viewportRef.current?.setSceneObjectMode(m), onAlign: (op, target) => viewportRef.current?.alignSceneObject(op, target),
                targets: alignTargets, target: alignTarget, onTarget: setAlignTarget, onRemove: () => viewportRef.current?.removeSelectedSceneObject(),
              } : null}
            />
          )}

          {hasSession && !sessionLoading && physicsOpen && (
            <PhysicsPanel diameterCm={physicsDiameterCm} onDiameterCm={setPhysicsDiameterCm} bounce={physicsBounce} onBounce={setPhysicsBounce} count={physicsCount}
              placing={physicsPlacing} onPlace={() => setPhysicsPlacing(v => !v)}
              onDrop={() => viewportRef.current?.physicsDrop(physicsDiameterCm / 100, physicsBounce)}
              onClear={() => viewportRef.current?.physicsClear()}
              onClose={() => { setPhysicsOpen(false); setPhysicsPlacing(false); viewportRef.current?.physicsClear() }} />
          )}

          {hasSession && !sessionLoading && (
            <ViewportHud onView={preset => viewportRef.current?.setStandardView(preset)} metersPerPixel={metersPerPixel}
              gridLabel={showGrid ? t('view.gridCell', { cell: gridLength(sceneLook.grid.cellM, fmt.number), major: gridLength(sceneLook.grid.cellM * sceneLook.grid.majorEvery, fmt.number) }) : null}
              getQuaternion={() => viewportRef.current?.getCameraQuaternion() ?? null}
              onOrient={dir => viewportRef.current?.orientView(dir)}
              onOrbit={(a, e) => viewportRef.current?.orbitBy(a, e)}
              onHome={() => viewportRef.current?.resetCamera()}
              colorMode={sabanaVisible ? 'deviation' : colorMode} mvThreshold={mvThreshold}
              statusFractions={certifyState?.report_metrics?.witnesses?.status_fraction ?? null}
              deviationRange={sabanaVisible && sabanaMetrics ? { maxMm: (sabanaMetrics.tolerance_mm || 15) * 3, toleranceMm: sabanaMetrics.tolerance_mm || 15 } : null}
              showEdges={showCertifyKit} readouts={readouts}
              onCalibrate={id => { const d = viewportRef.current?.getDistance(id); if (d) { setCalibrationDraft({ kind: 'distance', readoutId: id, ...d }); setCalibrationOpen(true) } }}
              onOpenCalibration={() => { setCalibrationDraft(null); setCalibrationOpen(true) }}
              calibration={calibrationSummary ? { state: calibrationSummary.state, factor: calibrationSummary.scale_factor,
                                                  axes: calibrationSummary.mode_used === 'axes' ? calibrationSummary.axes.factors : null,
                                                  worstDifferenceM: calibrationSummary.verification?.worst_difference_m ?? null } : null}
              banner={pipelineActiveHere && (viewerHasCloud || sessionHasPublishedCloud) && currentStage ? (
                <Banner glass compact tone="brand" title={t('pipeline.runningBanner', { stage: currentStage.label })} action={<Button size="sm" variant="ghost" onClick={handlePipelineCancel}>{t('common.cancel')}</Button>}>
                  {currentStage.message} {fmt.percent(currentStage.pct / 100)}
                </Banner>
              ) : undefined} />
          )}

          {flythroughOpen && <SyncPlayer sessionId={flythroughOpen} viewportRef={viewportRef} onClose={() => setFlythroughOpen(null)} />}

          {loadingOverlay && <SplashOverlay title={t('splash.loadingSession')} status={sessionLoading || undefined} />}
          {pipelineActiveHere && !viewerHasCloud && !sessionHasPublishedCloud && (
            <SplashOverlay title={t('splash.building')} status={currentStage ? `${currentStage.label}: ${currentStage.message}` : t('splash.initializing')} stages={pipelineStages} />
          )}
          {!hasSession && !sessionLoading && <WelcomeScreen connected={connected} sessionCount={sessions.length} onConnect={connectToServer} />}
        </div>

        {layout.state.dockOpen && (
          <div className="stac-main__dock">
            <BottomDock
              tabs={[
                { id: 'console', label: t('dock.console'), icon: <Terminal aria-hidden /> },
                { id: 'jobs', label: t('dock.jobs'), icon: <ListTodo aria-hidden />, badge: pipelineRunning ? 1 : undefined },
                { id: 'timeline', label: t('dock.timeline'), icon: <Clock aria-hidden /> },
                { id: 'report', label: t('dock.report'), icon: <BarChart3 aria-hidden /> },
                { id: 'evaluation', label: t('dock.evaluation'), icon: <FlaskConical aria-hidden /> },
              ]}
              panels={{
                console: <ConsolePanel logs={consoleLogs} />,
                jobs: <JobsPanel pipeline={pipelineRunning ? { session_id: pipelineRunning.session_id, status: pipelineRunning.status, stages: pipelineStages } : null} onCancelPipeline={handlePipelineCancel}
                  resume={resumeBusy ? resumeProgress : null} extracting={extractingSessions} tasks={activeTasks} />,
                timeline: <ScansPanel variant="timeline" sessionId={activeSession} scans={projectScans} activeScanKey={activeScanTab}
                  onOpenScan={sc => activeSession && openScanFromTree(activeSession, sc)} onSetReference={sc => activeSession && setReference(activeSession, sc)} />,
                report: activeSession && bimModels.length > 0 && segments.length > 0
                  ? <DeviationOverlay sessionId={activeSession} viewportRef={viewportRef} onClose={() => layout.setDockOpen(false)} />
                  : <EmptyState compact icon={<BarChart3 aria-hidden />} title={t('report.emptyTitle')} description={t('report.emptyDesc')} />,
                evaluation: <EvaluationPanel />,
              }} />
          </div>
        )}
      </main>

      <div className="stac-app__inspector">
        {layout.state.inspectorOpen && (
          <Inspector
            tabs={[
              { id: 'properties', label: t('inspector.properties') },
              { id: 'acta', label: t('inspector.acta'), badge: storedEpochs > 0 ? storedEpochs : undefined },
              { id: 'assistant', label: t('assistant.title') },
            ]}
            panels={{
              properties: <PropertiesPanel sessionId={activeSession} scanLabel={activeScanRow ? `${activeScanRow.date} ${activeScanRow.label}` : null} isReference={!!activeScanRow?.is_reference}
                pointCount={pointCount} fps={fps} epoch={epoch} storedEpochs={storedEpochs} segments={segments} selected={selectedSegment} unsegmentedCount={unsegmentedCount} bimCount={bimModels.length} hasSabana={hasSabanaResult} />,
              acta: (
                <div className="stac-inspector__stack">
                  {activeSession ? (
                    <CertifyKitPanel session={activeSession} token={token} viewport={viewportRef.current} onStatus={setStatusMessage} refreshKey={certifyRefresh}
                      colorMode={colorMode} onColorMode={setColorMode} mvThreshold={mvThreshold} onMvThreshold={setMvThreshold}
                      mvHide={mvHide} onMvHide={setMvHide} hasWitness={hasWitness} onStateLoaded={setCertifyState} />
                  ) : <EmptyState icon={<FileCheck2 aria-hidden />} title={t('properties.noSession')} description={t('welcome.pickSession')} />}
                  {sabanaFullMeta && activeSession && <BIMAnalysisPanel meta={sabanaFullMeta} sessionId={activeSession} />}
                </div>
              ),
              assistant: <AssistantPanel sessionId={activeSession} scanKey={activeScanTab} viewport={viewportRef} userKey={user?.username ?? null} getUiState={getUiState} onVlmStatus={setVlmStatus} onFocusObjects={focusSegments} onVolumesChanged={refreshVolumes} onViewerAction={runViewerAction} />,
            }} />
        )}
      </div>

      <div className="stac-app__status">
        <StatusBar serverAlive={serverAlive} connected={connected} sessionLabel={activeSession}
          scanLabel={activeScanRow ? `${activeScanRow.date} ${activeScanRow.label}` : null}
          message={statusMessage} epoch={epoch} storedEpochs={storedEpochs} cursor={cursor}
          job={pipelineActiveHere && currentStage ? currentStage.label : null}
          jobPct={pipelineActiveHere && currentStage ? currentStage.pct : null}
          points={pointCount} fps={fps} consoleOpen={consoleOpen} onToggleConsole={() => setConsoleOpen(!consoleOpen)} />
      </div>

      <PipelineDialog open={pipelineDialogOpen} sessionId={pipelineDialogSession} scans={scansList} selected={selectedScans}
        onToggleScan={(key, on) => setSelectedScans(prev => (on ? [...prev, key] : prev.filter(k => k !== key)))}
        replace={pipelineReplace} onReplace={setPipelineReplace}
        segment={pipelineSegment} onToggleSegment={(key, on) => setPipelineSegment(prev => (on ? [...prev.filter(k => k !== key), key] : prev.filter(k => k !== key)))}
        rebuildingKeys={(() => { const pr = pipelineRunning; const active = !!(pr && (pr.status === 'running' || pr.status === 'queued') && pr.session_id === pipelineDialogSession); return active ? (pr!.scans || scansList.map(s => s.key)) : [] })()}
        onCancel={() => setPipelineDialogOpen(false)} onRun={handlePipelineRun} />

      {adminOpen && <AdminPage onClose={() => setAdminOpen(false)} />}

      {(callTarget || incomingCall) && user && (
        <WebRTCCall wsRef={teamWsRef} userId={user.id} callTarget={callTarget} incomingCall={incomingCall}
          onClose={() => { setCallTarget(null); setIncomingCall(null) }} onIncomingHandled={() => setIncomingCall(null)} />
      )}

      {fuseSession && (
        <FuseScansModal project={fuseSession} scans={fuseSession === activeSession && projectScans.length ? projectScans : (allProjectScans[fuseSession] || [])}
          onClose={() => setFuseSession(null)} onStatus={setStatusMessage}
          onFused={async () => {
            const sid = fuseSession
            if (sid !== activeSession) { loadAllProjectScans([sid]); return }
            // the loaded project opens its new fused scan as a tab
            const scans = await loadProjectScans(sid)
            const fused = scans.filter(s => s.kind === 'fused').slice(-1)[0]
            if (fused) activateScan(sid, { key: fused.key, label: fused.label, date: fused.date, kind: 'fused' })
          }} />
      )}

      {resumeDialog && <ResumeDialog incomplete={resumeDialog.incomplete} complete={resumeDialog.complete} busy={resumeBusy} progress={resumeProgress} onResume={resumeRun} onCancelDelete={resumeCancel} />}

      {interactiveSessionId && (
        <SegmentationManager sessionId={interactiveSessionId} onClose={closeSegmentationManager}
          onUpdate={async () => { if (!activeSession) return; try { await reloadSegments(activeSession); viewportRef.current?.refreshSegmentOBBs(activeSession) } catch { /* silent */ } }}
          onSuccess={newInstances => { report(t('segmentation.propagated', { n: newInstances.length }), 'ok'); setInteractiveSessionId(null); if (activeSession) viewportRef.current?.sendCommand({ type: 'refresh_segments', session_id: activeSession }) }} />
      )}

      <Dialog open={compareDialogOpen} size="sm" title={t('deviation.compareTitle')} icon={<Scale aria-hidden />} tone="measure" onClose={() => setCompareDialogOpen(false)}
        footer={<><Button onClick={() => setCompareDialogOpen(false)}>{t('common.cancel')}</Button><Button variant="primary" onClick={runComparison} data-autofocus>{t('deviation.runComparison')}</Button></>}>
        <p className="stac-dialog__message">{t('deviation.compareDesc')}</p>
        <Field label={t('deviation.registration')} inline>
          <SegmentedControl<'auto' | 'manual'> ariaLabel={t('deviation.registration')} value={useManualAlignment ? 'manual' : 'auto'} onChange={v => setUseManualAlignment(v === 'manual')}
            options={[{ value: 'auto', label: t('deviation.registrationAuto') }, { value: 'manual', label: t('deviation.registrationManual') }]} />
        </Field>
      </Dialog>

      <ObjectLibraryDialog open={showObjectLibrary} items={objectLibrary} onClose={() => setShowObjectLibrary(false)}
        onPick={async it => {
          if (!activeSession) return
          try {
            const r = await fetch('/api/objects/scene/add', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: activeSession, name: it.name, url: it.url }) })
            const d = await r.json()
            if (d.ok) {
              setShowObjectLibrary(false)
              await viewportRef.current?.placeSceneObject(d.object)
              // a collection object (the reference human…) stands ON the floor, y = 0 (USER 2026-10-01)
              if (it.source === 'collection') viewportRef.current?.alignSceneObject('floor')
            }
          } catch { report(t('library.addFailed'), 'err') }
        }} />


      <AutosegmentDialog open={autosegOpen} sessionId={activeSession} state={autosegState} loading={autosegLoading}
        scans={projectScans.filter((sc: any) => sc.kind !== 'fused').map((sc: any) => ({ key: sc.key, label: `${sc.date} ${sc.label}` }))}
        selectedScans={autosegScans}
        onToggleScan={(key, on) => setAutosegScans(prev => (on ? [...prev.filter(k => k !== key), key] : prev.filter(k => k !== key)))}
        onClose={() => setAutosegOpen(false)} onRun={runAutosegment} />

      <GenreconDialog open={genreconOpen} sessionId={activeSession} scanLabel={activeScanTab} defaultConfidence={confidenceThreshold}
        onClose={() => setGenreconOpen(false)} onRun={runGenrecon} />

      {activeSession && (
        <ReferenceDialog open={referenceDialog !== null} sessionId={activeSession} scanKey={activeScanTab} initialMode={referenceDialog ?? 'distance'}
          onClose={() => setReferenceDialog(null)} />
      )}
      {activeSession && (
        <CalibrationDialog open={calibrationOpen} sessionId={activeSession} scanKey={activeScanTab} draft={calibrationDraft}
          otherScans={projectScans.filter((sc: any) => sc.key !== activeScanTab && sc.kind !== 'fused').map((sc: any) => ({ key: sc.key, label: sc.label || sc.key }))}
          onClose={() => {
            // a draft that was not stored leaves the scene with the window
            if (calibrationDraft?.kind === 'plane') viewportRef.current?.removeDistance(calibrationDraft.readoutId)
            setCalibrationOpen(false); setCalibrationDraft(null)
          }}
          onMeasure={kind => {
            setCalibrationOpen(false); setCalibrationDraft(null); setCalibrating(kind)
            setActiveTool(kind === 'distance' ? 'measure-distance' : 'measure-plane')
            setStatusMessage(tt(kind === 'distance' ? 'calibration.measureStatus' : 'calibration.planeStatus'))
          }}
          onChanged={drawCalibrationMarks}
          onStored={id => { viewportRef.current?.removeDistance(id); setCalibrationDraft(null) }}
          onFrame={pts => {
            const lo = [0, 1, 2].map(k => Math.min(...pts.map(q => q[k]))), hi = [0, 1, 2].map(k => Math.max(...pts.map(q => q[k])))
            const pad = Math.max(0.3, 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
            viewportRef.current?.frameBox(lo.map(v => v - pad), hi.map(v => v + pad))
          }} />
      )}

      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} commands={commands} />
      <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <SceneLookDialog open={sceneLookOpen} onClose={() => setSceneLookOpen(false)} look={sceneLook} onChange={setSceneLook} />
      {appDialog}
    </div>
  )
}

export default App
