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
import Viewport, { ViewportHandle, SegmentInstance } from './components/Viewport'
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
  Clock, Scale, BarChart3, Home, Camera, SlidersHorizontal, Sparkles, Undo2, Brush, Layers, Star, AlertTriangle, Terminal,
  ListTodo, FileCheck2, Puzzle, Wrench, PanelLeftClose, PanelLeftOpen,
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
import { SettingsDialog } from './layout/SettingsDialog'
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
import { ViewportHud, type ReadoutAnchor } from './components/viewport/ViewportHud'
import { SplashOverlay } from './components/viewport/SplashOverlay'
import { WelcomeScreen } from './components/viewport/WelcomeScreen'
import { tokenColor } from './components/viewport/palette'
import { SessionsPanel } from './features/SessionsPanel'
import { InstancesPanel } from './features/InstancesPanel'
import { ScansPanel } from './features/ScansPanel'
import { PipelineDialog } from './features/PipelineDialog'
import { ResumeDialog } from './features/ResumeDialog'
import { ObjectLibraryDialog } from './features/ObjectLibraryDialog'
import { CorrectionDialog } from './features/CorrectionDialog'
import { MeshingDialog } from './features/MeshingDialog'
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

type Tool = 'navigate' | 'measure-distance' | 'measure-angle' | 'section-box' | 'align' | 'erase'

function App() {
  // auth first: token feeds the correction handlers' deps below (TDZ)
  const { user, token, loading: authLoading, logout } = useAuth()
  const { confirmDanger, dialogElement: appDialog } = useConfirmDialog()
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [activeSession, setActiveSession] = useState<string | null>(null)
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [activeTool, setActiveTool] = useState<Tool>('navigate')
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
  //  Correction Analysis (USER 2026-09-06, replaces Perfect in the
  // toolbar): user marks the segments with the parallel-copies error; the
  // algorithm does the rest. Every epoch stays and is selected in the modal.
  // Object library (user 2026-08-31): collection + every project GLB,
  // insertable as scene REFERENCES (delete never touches the source)
  const [showObjectLibrary, setShowObjectLibrary] = useState(false)
  const [objectLibrary, setObjectLibrary] = useState<Array<{ source: string; session?: string; name: string; url: string; size_mb: number }>>([])
  const [sceneObjSel, setSceneObjSel] = useState<number | null>(null)
  const [placedObjects, setPlacedObjects] = useState<Array<{ id: number; name: string; visible: boolean }>>([])
  const [alignTargets, setAlignTargets] = useState<Array<{ key: string; label: string }>>([])
  const [alignTarget, setAlignTarget] = useState('')
  const [showCorrectionModal, setShowCorrectionModal] = useState(false)
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
  const [correctionSelected, setCorrectionSelected] = useState<Set<number>>(new Set())
  const [correctionRunning, setCorrectionRunning] = useState(false)
  const [correctionState, setCorrectionState] = useState<any>(null)
  const [pendingSession, setPendingSession] = useState<string | null>(null)
  const [correctionEpochs, setCorrectionEpochs] = useState<{ epoch: number; live: boolean; potree: boolean }[]>([])
  const refreshCorrectionStatus = useCallback(async (sid: string) => {
    try {
      const r = await fetch(`/api/correction/state/${sid}`)
      if (r.ok) {
        const st = await r.json()
        setCorrectionState(st)
        // the state carries the epochs the session holds, so the selector is
        // populated on load and not only after a selection (USER 2026-09-16)
        if (Array.isArray(st?.epochs)) setCorrectionEpochs(st.epochs)
        if (st?.status === 'applied') setPendingSession(sid)
        else if (pendingSessionRef.current === sid) setPendingSession(null)
      }
    } catch { /* non-fatal */ }
  }, [])
  const pendingSessionRef = useRef<string | null>(null)
  useEffect(() => { pendingSessionRef.current = pendingSession }, [pendingSession])
  // keep the pending session's status honest even while ANOTHER session is
  // active (USER 2026-09-06: the banner survives session switches)
  useEffect(() => {
    if (!pendingSession || pendingSession === activeSession) return
    const iv = setInterval(() => refreshCorrectionStatus(pendingSession), 15000)
    return () => clearInterval(iv)
  }, [pendingSession, activeSession, refreshCorrectionStatus])
  // Correction panel state (USER 2026-09-08 redesign: keyframe-based flow,
  // no chunk gizmo). The last run's full report (also when rejected), the
  // live stage progress, the floor model, the ledger and the stale badges.
  const [correctionReport, setCorrectionReport] = useState<any>(null)
  const [correctionProgress, setCorrectionProgress] = useState<{ pct: number; detail: string } | null>(null)
  const [correctionOverrideScale, setCorrectionOverrideScale] = useState(false)
  const [correctionLedger, setCorrectionLedger] = useState<any[] | null>(null)
  const [correctionArtifacts, setCorrectionArtifacts] = useState<any[] | null>(null)
  // shared epoch-selection handlers — used by the  modal AND the banner
  // (USER 2026-09-06: an applied correction MUST be visible).
  // EPOCHS ARE SELECTED, NEVER APPROVED (USER 2026-09-16: "todas viven, solo
  // se seleccionan y la que se selecciona se muestra"). Approve deleted every
  // other epoch and Undo the current one, so the session could only hold two
  // states and one wrong click destroyed the other — pccr 2026-09-15 lost
  // epochs 2 and 3 to two accidental Undos.
  const refreshCorrectionEpochs = useCallback(async (sid?: string | null) => {
    const s = sid || pendingSessionRef.current || activeSession
    if (!s) return
    try {
      const r = await fetch(`/api/correction/epochs/${encodeURIComponent(s)}`)
      if (r.ok) setCorrectionEpochs((await r.json()).epochs || [])
    } catch { /* the list is a convenience; a failure leaves the last one */ }
  }, [activeSession])
  const selectCorrectionEpoch = useCallback(async (epoch: number) => {
    const sid = pendingSessionRef.current || activeSession
    if (!sid) return
    setCorrectionRunning(true)
    setStatusMessage(tt('correction.selecting', { epoch }))
    try {
      const headers: HeadersInit = { 'Content-Type': 'application/json' }
      if (token) headers['Authorization'] = `Bearer ${token}`
      const r = await fetch('/api/correction/select', {
        method: 'POST', headers,
        body: JSON.stringify({ session_id: sid, epoch }),
      })
      setStatusMessage(r.ok ? tt('correction.selected', { epoch })
                            : tt('correction.selectFailed'))
    } catch { setStatusMessage(tt('correction.selectFailed')) }
    setCorrectionRunning(false)
    refreshCorrectionStatus(pendingSessionRef.current || activeSession!)
    refreshCorrectionEpochs(pendingSessionRef.current || activeSession!)
  }, [activeSession, refreshCorrectionStatus, refreshCorrectionEpochs, token])
  // pending state must surface ALWAYS (USER 2026-09-06: the banner appears
  // as soon as the potree loads and stays — across session switches too —
  // while the session holds more than one epoch). A long-running
  // correction POST can outlive the browser request, so the truth is
  // POLLED from the backend, not inferred from the fetch result.
  useEffect(() => {
    if (!activeSession) return
    refreshCorrectionStatus(activeSession)
    const iv = setInterval(() => refreshCorrectionStatus(activeSession), 15000)
    return () => clearInterval(iv)
  }, [activeSession, refreshCorrectionStatus, token])
  // ── Multi-scan project, VS-Code-style TABS (USER 2026-09-06): one tab
  // per open scan; the ACTIVE tab is the scan being viewed AND worked on
  // (segmentation, brush, corrections, meshing, chat). Closing a tab loses
  // nothing; the scans list reopens it (double-click). The composition
  // reference is a  on the list row (one per project).
  type ScanTab = { key: string; label: string; date: string; kind: 'scan' | 'fused' }
  const [projectScans, setProjectScans] = useState<any[]>([])
  // scans of EVERY project (tree children in the Projects list); the
  // loaded project's entry is refreshed together with projectScans
  const [allProjectScans, setAllProjectScans] = useState<Record<string, any[]>>({})
  const [scanTabs, setScanTabs] = useState<ScanTab[]>([])
  const [activeScanTab, setActiveScanTab] = useState<string | null>(null)
  const [showFuseModal, setShowFuseModal] = useState(false)
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
  const loadCorrectionLedger = useCallback(async (sid: string) => {
    try {
      const r = await fetch(`/api/correction/ledger/${sid}`)
      if (r.ok) setCorrectionLedger((await r.json()).entries || [])
    } catch { /* non-fatal */ }
  }, [])
  const loadCorrectionArtifacts = useCallback(async (sid: string) => {
    try {
      const r = await fetch(`/api/correction/artifacts/${sid}`)
      if (r.ok) setCorrectionArtifacts((await r.json()).artifacts || [])
    } catch { /* non-fatal */ }
  }, [])
  // per-stage progress while a correction runs (task_type "correction")
  useEffect(() => {
    if (!correctionRunning || !activeSession) { setCorrectionProgress(null); return }
    const iv = setInterval(async () => {
      try {
        const d = await fetch(`/api/tasks/${activeSession}`).then(r => r.json())
        const t = (d.tasks || []).find((x: any) => x.task_type === 'correction')
        if (t) setCorrectionProgress({ pct: t.pct || 0, detail: t.detail || t.label || '' })
      } catch { /* keep last */ }
    }, 2000)
    return () => clearInterval(iv)
  }, [correctionRunning, activeSession])
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
  const [pointCount, setPointCount] = useState(0)
  const [sessionLoading, setSessionLoading] = useState<string | null>(null)
  const [fps, setFps] = useState(0)
  const [consoleLogs, setConsoleLogs] = useState<{ ts: string; level: string; msg: string }[]>([])
  const viewportRef = useRef<ViewportHandle>(null)
  // Timestamp (ms) of the last time a cloud actually loaded into the viewer (point
  // count > 0). Used by the pipeline-done handler to tell "the cloud arrived on this
  // live socket" from "WS reconnected mid-pipeline  potree_ready went to a dead socket".
  const lastCloudLoadAtRef = useRef(0)
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
  useEffect(() => {
    let cancelled = false
    const reconcile = async () => {
      try {
        const r = await fetch('/api/pipelines/active')
        if (!r.ok || cancelled) return
        const live = ((await r.json())?.pipelines ?? {}) as Record<string, unknown>
        setPipelineRunning(prev => {
          if (!prev) return prev
          if (Object.prototype.hasOwnProperty.call(live, prev.session_id ?? '')) return prev
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
    setTsdfRunning(false)
    setShapeRunning(false)
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
  const [shapeMeshes, setShapeMeshes] = useState<MeshListItem[]>([])
  const [tsdfMeshes, setTsdfMeshes] = useState<MeshListItem[]>([])

  const [shapeRunning, setShapeRunning] = useState(false)
  // Ref-based busy guard. The `disabled` prop on the Start button only gates
  // clicks AFTER React commits the next render — between the click and that
  // commit (a few ms in the worst case, more under load) the DOM button is
  // still clickable. A bouncy mouse, an over-eager screen reader, or an
  // intermediary retry can all sneak a second click through. The ref is set
  // synchronously, so any second entry within the same JS turn is rejected.
  const shapeBusyRef = useRef(false)
  const [, setShapeStatus] = useState<Record<number, { has_pkl: boolean; has_mesh: boolean }>>({})

  // Live progress for the in-flight shape run
  type ShapeInstanceProgress = {
    id: number
    phase: string  // captioning | exporting_pkl | pkl_ready | reconstructing | done | error
    elapsed?: number
    error?: string
    mesh?: string
  }
  const [, setShapeProgress] = useState<Record<number, ShapeInstanceProgress>>({})
  const [shapeOverall, setShapeOverall] = useState<{ phase: string; total?: number; done?: number }>({ phase: 'idle' })

  const refreshShapeStatus = useCallback(async () => {
    if (!activeSession) return
    try {
      const res = await fetch(`/api/segmentation/shape/status/${activeSession}`)
      const data = await res.json()
      if (data.ok) {
        const statusMap: Record<number, { has_pkl: boolean; has_mesh: boolean }> = {}
        for (const inst of data.instances) {
          statusMap[inst.id] = { has_pkl: inst.has_pkl, has_mesh: inst.has_mesh }
        }
        setShapeStatus(statusMap)
      }
    } catch { /* ignore */ }
  }, [activeSession])

  // Refresh generated-mesh list for the sidebar (mirrors what Viewport fetches —
  // shared endpoint). Preserves visibility for instances that already exist
  // so a refresh after generation doesn't reset user-toggled hides.
  const refreshShapeMeshList = useCallback(async (sessionId: string) => {
    try {
      const res = await fetch(`/api/segmentation/shape/list/${sessionId}`)
      if (!res.ok) { setShapeMeshes([]); return }
      const data = await res.json()
      const items = (data?.shapes || []) as Array<{
        folder: string
        meta: { instance_id?: number; label?: string }
      }>
      setShapeMeshes(prev => {
        const prevVis = new Map(prev.map(m => [m.instanceId, m.visible]))
        return items
          .filter(s => typeof s.meta?.instance_id === 'number')
          .map(s => ({
            instanceId: s.meta.instance_id as number,
            label: (s.meta.label as string) || `object_${s.meta.instance_id}`,
            folder: s.folder,
            visible: prevVis.get(s.meta.instance_id as number) ?? true,
          }))
      })
    } catch { /* ignore */ }
  }, [])

  // Same for TSDF mesh list.
  const refreshTsdfMeshList = useCallback(async (sessionId: string) => {
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
          // Scene meshes start visible (auto-loaded). Per-instance meshes are
          // lazy: every refresh follows a viewer reset (reloadTsdf / session
          // load) that dropped them back to not-loaded, so their checkbox
          // always resets to unchecked — checking it triggers the on-demand
          // download in the Viewport (setTsdfVisibility).
          visible: typeof s.meta?.instance_id === 'number'
            ? false
            : (prevVis.get(s.folder) ?? true),
        }))
      })
    } catch { /* ignore */ }
  }, [])

  // Poll progress while running
  useEffect(() => {
    if (!shapeRunning || !activeSession) return
    let alive = true
    const tick = async () => {
      try {
        const res = await fetch(`/api/segmentation/shape/progress/${activeSession}`)
        const data = await res.json()
        if (!alive) return
        if (data.ok) {
          const map: Record<number, ShapeInstanceProgress> = {}
          for (const inst of (data.instances || [])) map[inst.id] = inst
          setShapeProgress(map)
          setShapeOverall(data.overall || { phase: 'idle' })
          // When backend is done (or errored), refresh has_pkl/has_mesh and stop polling
          if (data.overall?.phase === 'done' || data.overall?.phase === 'error') {
            await refreshShapeStatus()
            // Reload meshes into the viewport so the new .glb files appear
            // automatically next to the point cloud.
            if (activeSession) {
              try { await viewportRef.current?.reloadShapes(activeSession) } catch { /* ignore */ }
              await refreshShapeMeshList(activeSession)
            }
            setShapeRunning(false)
          }
        }
      } catch { /* ignore transient errors */ }
    }
    tick()
    const id = window.setInterval(tick, 1500)
    return () => { alive = false; window.clearInterval(id) }
  }, [shapeRunning, activeSession, refreshShapeStatus, refreshShapeMeshList])

  // ── TSDF export state (parallel to ShapeR — same UX, different backend) ──
  const [showTsdfModal, setShowTsdfModal] = useState(false)
  const [tsdfRunning, setTsdfRunning] = useState(false)
  const [tsdfSelected, setTsdfSelected] = useState<Set<number>>(new Set())
  const [tsdfStatus, setTsdfStatus] = useState<Record<number, { has_mesh: boolean }>>({})
  type TsdfInstanceProgress = {
    id: number
    phase: string  // pending | starting | integrating | extracting | done | error
    elapsed?: number
    error?: string
    mesh?: string
  }
  const [tsdfProgress, setTsdfProgress] = useState<Record<number, TsdfInstanceProgress>>({})
  const [tsdfOverall, setTsdfOverall] = useState<{ phase: string; total?: number; done?: number }>({ phase: 'idle' })

  // Whole-scene TSDF (no instance filtering). Tracked independently from
  // per-instance progress so both can be displayed in the same modal.
  // (scene-level jobs no longer have UI buttons — user 2026-08-29 — but the
  // polling effect still tracks them so an externally launched run finishes
  // cleanly; only the setters are needed.)
  const [tsdfSceneRunning, setTsdfSceneRunning] = useState(false)
  const [, setTsdfScene] = useState<{ phase: string; elapsed?: number; mesh?: string; error?: string }>({ phase: 'idle' })
  const [poissonSceneRunning, setPoissonSceneRunning] = useState(false)
  const [, setPoissonScene] = useState<{ phase: string; elapsed?: number; mesh?: string; error?: string }>({ phase: 'idle' })

  const refreshTsdfStatus = useCallback(async () => {
    if (!activeSession) return
    try {
      const res = await fetch(`/api/segmentation/tsdf/status/${activeSession}`)
      const data = await res.json()
      if (data.ok) {
        const statusMap: Record<number, { has_mesh: boolean }> = {}
        for (const inst of data.instances) statusMap[inst.id] = { has_mesh: inst.has_mesh }
        setTsdfStatus(statusMap)
      }
    } catch { /* ignore */ }
  }, [activeSession])

  const openTsdfModal = useCallback(async () => {
    setTsdfProgress({})
    setTsdfOverall({ phase: 'idle' })
    setShowTsdfModal(true)
    if (!activeSession) return
    try {
      const res = await fetch(`/api/segmentation/tsdf/status/${activeSession}`)
      const data = await res.json()
      if (data.ok) {
        const statusMap: Record<number, { has_mesh: boolean }> = {}
        const selectedIds = new Set<number>()
        for (const inst of data.instances) {
          statusMap[inst.id] = { has_mesh: inst.has_mesh }
          if (!inst.has_mesh) selectedIds.add(inst.id)
        }
        setTsdfStatus(statusMap)
        setTsdfSelected(selectedIds)
      }
    } catch { /* ignore */ }
  }, [activeSession])

  // Poll TSDF progress while running (per-instance OR whole-scene)
  useEffect(() => {
    if ((!tsdfRunning && !tsdfSceneRunning && !poissonSceneRunning) || !activeSession) return
    let alive = true
    const tick = async () => {
      try {
        const res = await fetch(`/api/segmentation/tsdf/progress/${activeSession}`)
        const data = await res.json()
        if (!alive) return
        if (data.ok) {
          const map: Record<number, TsdfInstanceProgress> = {}
          for (const inst of (data.instances || [])) map[inst.id] = inst
          setTsdfProgress(map)
          setTsdfOverall(data.overall || { phase: 'idle' })
          const scene = data.scene || { phase: 'idle' }
          setTsdfScene(scene)
          const poisson = data.poisson || { phase: 'idle' }
          setPoissonScene(poisson)
          // Each completion block is gated on ITS running flag so a PERSISTED
          // 'done'/'error' phase from an earlier job can't re-fire every tick
          // (which reloaded scene.glb on a loop and hung the UI while another
          // job — e.g. Poisson — was the one actually running).
          // 'idle' while we think we're running = the backend has no such job
          // (restart wiped it, or it was another session's) — unstick the
          // button instead of blocking Meshing forever (user 2026-08-31,
          // observatorio: "no me deja hacer meshing").
          if (tsdfRunning && (!data.overall || data.overall.phase === 'idle')) {
            setTsdfRunning(false)
          }
          if (tsdfRunning && (data.overall?.phase === 'done' || data.overall?.phase === 'error')) {
            await refreshTsdfStatus()
            if (activeSession) {
              try { await viewportRef.current?.reloadTsdf?.(activeSession) } catch { /* ignore */ }
              await refreshTsdfMeshList(activeSession)
            }
            setTsdfRunning(false)
          }
          if (tsdfSceneRunning && (scene.phase === 'done' || scene.phase === 'error')) {
            if (scene.phase === 'done' && activeSession) {
              // Whole-scene TSDF wrote output/tsdf/scene/scene.glb — pull it
              // into the viewport (tsdf/list already includes the scene/ dir).
              try { await viewportRef.current?.reloadTsdf?.(activeSession) } catch { /* ignore */ }
              await refreshTsdfMeshList(activeSession)
            }
            setTsdfSceneRunning(false)
          }
          if (poissonSceneRunning && (poisson.phase === 'done' || poisson.phase === 'error')) {
            if (poisson.phase === 'done' && activeSession) {
              // Poisson wrote output/tsdf/scene_poisson/scene_poisson.glb — pull it
              // into the viewport (tsdf/list iterates all tsdf/ subfolders).
              try { await viewportRef.current?.reloadTsdf?.(activeSession) } catch { /* ignore */ }
              await refreshTsdfMeshList(activeSession)
            }
            setPoissonSceneRunning(false)
          }
        }
      } catch { /* ignore */ }
    }
    tick()
    const id = window.setInterval(tick, 1200)
    return () => { alive = false; window.clearInterval(id) }
  }, [tsdfRunning, tsdfSceneRunning, poissonSceneRunning, activeSession, refreshTsdfStatus, refreshTsdfMeshList])

  // Refresh shape/TSDF mesh lists when the active session changes (or clear
  // them on session unload). The Viewport already auto-loads the GLBs into
  // the scene; this keeps the sidebar list in sync with what's on disk.
  useEffect(() => {
    if (!activeSession) {
      setShapeMeshes([])
      setTsdfMeshes([])
      return
    }
    refreshShapeMeshList(activeSession)
    refreshTsdfMeshList(activeSession)
  }, [activeSession, refreshShapeMeshList, refreshTsdfMeshList])

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

  // Dismiss loading overlay when points arrive (session load only, not refresh)
  const prevPointCount = useRef(0)
  useEffect(() => {
    if (prevPointCount.current === 0 && pointCount > 0 && sessionLoading) {
      // If session has BIM, don't dismiss yet — wait for BIM load + sábana check
      const sess = sessions.find(s => s.id === activeSession)
      if (!sess?.hasBim) {
        setSessionLoading(null)
      }
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
        const tasks = data.tasks as { label: string; pct: number; detail: string }[]
        if (tasks && tasks.length > 0) {
          const t = tasks[0]  // show the first active task
          const msg = t.detail ? `${t.label} — ${t.detail}` : t.label
          // Don't set sessionLoading when pipeline is running — pipeline panel already shows progress
          const pr = pipelineRunningRef.current
          const pipelineActive = pr && (pr.status === 'running' || pr.status === 'queued') && pr.session_id === activeSession
          if (!pipelineActive) {
            setSessionLoading(msg)
          }
          // Keep polling while tasks are active
          timer = setTimeout(poll, 3000)
        } else {
          // Tasks done — if overlay was showing a server task, dismiss it
          if (sessionLoading && pointCount > 0) {
            setSessionLoading(null)
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

  const applyFloorLevel = useCallback(async (
    sessionId: string, mode: 'auto' | 'explicit' | 'auto_if_needed', instanceId?: number
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

  // Floory=0 on session load: once the cloud is actually in the viewer
  // (pointCount > 0), detect an un-leveled segmented floor and fix it —
  // the server no-ops when it is already level. Once per session.
  useEffect(() => {
    if (!activeSession || pointCount <= 0) return
    if (floorLevelCheckedRef.current === activeSession) return
    floorLevelCheckedRef.current = activeSession
    applyFloorLevel(activeSession, 'auto_if_needed')
    refreshFloorLevel(activeSession)
  }, [activeSession, pointCount, applyFloorLevel, refreshFloorLevel])

  const handleReconstruct = useCallback(async (sessionId: string) => {
    setPipelineDialogSession(sessionId)
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

  const handlePipelineRun = useCallback(() => {
    if (!pipelineDialogSession) return
    if (selectedScans.length === 0) return
    // Load the session immediately so the websocket connects for progress
    setActiveSession(pipelineDialogSession)
    setSelectedSession(pipelineDialogSession)
    setPipelineDialogOpen(false)
    setPipelineRunning({ session_id: pipelineDialogSession, status: 'queued', current_stage_idx: -1, stages: [], scans: [...selectedScans] })
    setTimeout(() => {
      viewportRef.current?.sendCommand({
        type: 'run_pipeline',
        session_id: pipelineDialogSession,
        replace: pipelineReplace,
        scans: selectedScans,
      })
      setStatusMessage(tt('pipeline.started', { session: pipelineDialogSession, n: selectedScans.length }))
    }, 500)
  }, [pipelineDialogSession, pipelineReplace, selectedScans])

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
        viewportRef.current?.sendCommand({ type: 'load_session', session_id: sid })
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
        setTimeout(() => { prefsLoadedRef.current = true }, 300)
      })
      .catch(() => {
        setPointSize(5.0)
        setConfidenceThreshold(0.0)
        setPointBudget(10_000_000)
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
        body: JSON.stringify({ pointSize, confidenceThreshold, pointBudget })
      }).catch(() => {})
    }, 500)
    return () => { if (prefsSaveTimer.current) clearTimeout(prefsSaveTimer.current) }
  }, [pointSize, confidenceThreshold, pointBudget])  

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
  const { t, fmt } = useI18n()
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
  const [cursor, setCursor] = useState<{ x: number; y: number; z: number } | null>(null)
  const [readouts, setReadouts] = useState<ReadoutAnchor[]>([])
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
  }, [activeSession, certifyRefresh, correctionState])
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
      report(d.ok ? t('brush.undone', { n: fmt.integer(d.restored || 0) }) : t('brush.nothingToUndo'), d.ok ? 'ok' : 'warn')
    } catch { report(t('brush.undoFailed'), 'err') }
  }, [activeSession, report, t, fmt])

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
    await fetch('/api/segmentation/rename', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: activeSession, instance_id: seg.id, label: newLabel, old_label: seg.label.replace(/ #\d+$/, '') }),
    })
    setSegments(prev => prev.map(s => (s.key === seg.key ? { ...s, label: newLabel } : s)))
    viewportRef.current?.refreshSegmentOBBs(activeSession)
  }, [activeSession])

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

  const openCorrection = useCallback(() => {
    if (!activeSession) return
    setCorrectionSelected(new Set()); setCorrectionReport(null); setShowCorrectionModal(true)
    refreshCorrectionStatus(activeSession); loadCorrectionLedger(activeSession); loadCorrectionArtifacts(activeSession)
  }, [activeSession, refreshCorrectionStatus, loadCorrectionLedger, loadCorrectionArtifacts])

  const runCorrectionOp = useCallback(async (kind: 'objects' | 'floor' | 'revisit') => {
    if (!activeSession) return
    setCorrectionRunning(true)
    setCorrectionReport(null)
    setStatusMessage(kind === 'objects' ? t('correction.analyzing') : kind === 'floor' ? t('correction.aligningFloor') : t('correction.detecting'))
    try {
      const headers: HeadersInit = { 'Content-Type': 'application/json' }
      if (token) headers['Authorization'] = `Bearer ${token}`
      const url = kind === 'objects' ? '/api/correction/run' : kind === 'floor' ? '/api/correction/floor' : '/api/correction/revisit'
      const body = kind === 'objects'
        ? { session_id: activeSession, instance_ids: Array.from(correctionSelected), override_scale_check: correctionOverrideScale }
        : kind === 'floor' ? { session_id: activeSession, model: 'plane', keyframes: 'auto' } : { session_id: activeSession }
      const r = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) })
      const d = await r.json().catch(() => ({}))
      setCorrectionReport(d.report || null)
      if (r.ok && d.status === 'applied') setShowCorrectionModal(false)
      if (r.status === 409) report(t('correction.busy'), 'warn')
      else if (r.ok && d.status === 'applied') report(kind === 'revisit' ? t('correction.closuresApplied', { n: d.report?.solutions?.length || 0 }) : t('correction.applied'), 'ok')
      else if (r.ok) report(t('correction.rejected', { reason: d.report?.rejection_reason || t('correction.seeReport') }), 'warn')
      else report(t('correction.failed', { detail: typeof d.detail === 'string' ? d.detail : '' }), 'err')
    } catch { report(t('correction.failed', { detail: '' }), 'err') }
    setCorrectionRunning(false)
    if (kind === 'objects') setCorrectionOverrideScale(false)
    refreshCorrectionStatus(activeSession); loadCorrectionLedger(activeSession); loadCorrectionArtifacts(activeSession)
  }, [activeSession, token, correctionSelected, correctionOverrideScale, refreshCorrectionStatus, loadCorrectionLedger, loadCorrectionArtifacts, report, t])

  const runObjectMeshing = useCallback(async () => {
    if (!activeSession || shapeBusyRef.current) return
    shapeBusyRef.current = true
    setShapeRunning(true)
    setShapeProgress({})
    setShapeOverall({ phase: 'exporting_pkl', total: tsdfSelected.size, done: 0 })
    try {
      const res = await fetch('/api/segmentation/shape/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: activeSession, instance_ids: [...tsdfSelected], auto_reconstruct: true }) })
      const data = await res.json()
      if (data.ok) { report(t('meshing.objectStarted', { n: data.count }), 'ok'); if (!data.reconstructing) setShapeRunning(false) }
      else { report(t('meshing.objectError', { detail: data.detail || '' }), 'err'); setShapeRunning(false) }
    } catch (err: any) { report(t('meshing.objectError', { detail: err.message }), 'err'); setShapeRunning(false) }
    finally { shapeBusyRef.current = false }
  }, [activeSession, tsdfSelected, report, t])

  const runMesh = useCallback(async () => {
    if (!activeSession) return
    setTsdfRunning(true)
    setTsdfProgress({})
    setTsdfOverall({ phase: 'surface_fit', total: tsdfSelected.size, done: 0 })
    try {
      const res = await fetch('/api/segmentation/tsdf/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: activeSession, instance_ids: [...tsdfSelected] }) })
      const data = await res.json()
      if (data.ok) report(t('meshing.meshStarted', { n: data.count }), 'ok')
      else { report(t('meshing.meshError', { detail: data.detail || '' }), 'err'); setTsdfRunning(false) }
    } catch (err: any) { report(t('meshing.meshError', { detail: err.message }), 'err'); setTsdfRunning(false) }
  }, [activeSession, tsdfSelected, report, t])

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
        report(t('segmentation.refreshed', { n: data.instances?.length || 0 }), 'ok')
        if (Array.isArray(data.instances)) {
          setSegments(data.instances.map((inst: any) => ({
            key: inst.global_id || `${inst.label}_${inst.instance_id || inst.id}`,
            id: inst.instance_id || inst.id,
            label: `${inst.label}`,
            color: inst.color || tokenColor('--measure'),
            totalPoints: inst.total_points || 0,
            visible: true,
            excluded: inst.excluded || false,
          })))
        }
        viewportRef.current?.refreshSegmentOBBs(sid)
        refreshUnsegmentedCount(sid)
        await applyFloorLevel(sid, 'auto')
        refreshFloorLevel(sid)
        viewportRef.current?.sendCommandPreserveCamera({ type: 'load_session', session_id: sid })
      }
    } catch { /* silent */ }
    finally { setSessionLoading(null) }
  }, [interactiveSessionId, reloadSegments, refreshUnsegmentedCount, applyFloorLevel, refreshFloorLevel, report, t])

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
      { type: 'separator', id: 's2' },
      { id: 'console', label: t('dock.console'), icon: <Monitor aria-hidden />, shortcut: 'Mod+`', checked: consoleOpen, onSelect: () => setConsoleOpen(!consoleOpen) },
      { id: 'jobs', label: t('dock.jobs'), icon: <ListTodo aria-hidden />, checked: layout.state.dockOpen && layout.state.dockTab === 'jobs', onSelect: () => layout.openDock('jobs') },
      { type: 'separator', id: 's3' },
      { id: 'axes', label: t('menu.axes'), icon: <Axis3D aria-hidden />, checked: showAxes, onSelect: () => setShowAxes(v => !v) },
      { id: 'grid', label: t('menu.grid'), icon: <Grid3X3 aria-hidden />, checked: showGrid, onSelect: () => setShowGrid(v => !v) },
      { id: 'density', label: t('settings.density'), icon: <SlidersHorizontal aria-hidden />, hint: t(`settings.${layout.state.density}`), onSelect: () => setSettingsOpen(true) },
    ] },
    { id: 'tools', label: t('menu.tools'), disabled: !hasSession, entries: [
      { id: 'navigate', label: t('toolbar.navigate'), icon: <RotateCcw aria-hidden />, shortcut: 'V', checked: activeTool === 'navigate', onSelect: () => setActiveTool('navigate') },
      { id: 'resetview', label: t('toolbar.resetView'), icon: <Home aria-hidden />, shortcut: 'Home', onSelect: () => viewportRef.current?.resetCamera() },
      { type: 'separator', id: 's1' },
      { id: 'distance', label: t('toolbar.measureDistance'), icon: <Ruler aria-hidden />, shortcut: 'M', checked: activeTool === 'measure-distance', onSelect: () => setActiveTool('measure-distance') },
      { id: 'angle', label: t('toolbar.measureAngle'), icon: <TriangleRight aria-hidden />, shortcut: 'A', checked: activeTool === 'measure-angle', onSelect: () => setActiveTool('measure-angle') },
      { id: 'clearm', label: t('toolbar.clearMeasurements'), icon: <Trash2 aria-hidden />, onSelect: () => viewportRef.current?.clearMeasurements() },
      { type: 'separator', id: 's2' },
      { id: 'brush', label: t('toolbar.brush'), icon: <Brush aria-hidden />, checked: activeTool === 'erase', onSelect: () => setActiveTool(activeTool === 'erase' ? 'navigate' : 'erase') },
      { id: 'undobrush', label: t('brush.undo'), icon: <Undo2 aria-hidden />, onSelect: undoBrush },
      { id: 'align', label: t('toolbar.alignCloud'), icon: <Move aria-hidden />, shortcut: 'G', disabled: sabanaVisible, checked: activeTool === 'align', onSelect: () => setActiveTool(activeTool === 'align' ? 'navigate' : 'align') },
      { id: 'addobj', label: t('toolbar.addObject'), icon: <Package aria-hidden />, onSelect: openObjectLibrary },
      { type: 'separator', id: 's3' },
      { id: 'section', label: t('toolbar.sectionBox'), icon: <Scissors aria-hidden />, shortcut: 'X', checked: activeTool === 'section-box', onSelect: () => setActiveTool('section-box') },
      { id: 'resetsection', label: t('toolbar.resetSection'), icon: <Unlock aria-hidden />, onSelect: () => { viewportRef.current?.resetSectionBox(); setActiveTool('navigate') } },
      { type: 'separator', id: 's4' },
      { id: 'segmentation', label: t('instances.segmentation'), icon: <Crosshair aria-hidden />, onSelect: () => activeSession && openSegmentationManager(activeSession) },
      { id: 'meshing', label: t('instances.meshing'), icon: <Puzzle aria-hidden />, onSelect: openTsdfModal },
      { id: 'correction', label: t('instances.correction'), icon: <Wrench aria-hidden />, onSelect: openCorrection },
      { id: 'fuse', label: t('instances.fuse'), icon: <Layers aria-hidden />, disabled: projectScans.filter(sc => sc.kind !== 'fused').length < 2, onSelect: () => setShowFuseModal(true) },
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
  const loadingOverlay = sessionLoading && !pipelineActiveHere
  const pipelineStages: ProgressStage[] = (pipelineRunning?.stages || []).filter(s => s.enabled).map(s => ({ id: s.id, label: s.label, status: s.status, pct: s.pct, detail: s.message }))
  const currentStage = pipelineRunning?.stages[pipelineRunning.current_stage_idx]
  const epoch: number | null = certifyState?.epoch ?? correctionState?.epoch ?? null
  // how many OTHER epochs the session can be shown in — not a pending
  // verdict (USER 2026-09-16: nothing waits for approval any more)
  const storedEpochs: number = Math.max((certifyState?.epochs?.length ?? 0) - 1, 0)
  const selectedSegment = segments.find(s => s.id === selectedSegmentId) ?? null
  const hasSabanaResult = !!sessions.find(s => s.id === activeSession)?.hasSabana

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
          sessions={sessions.map(s => ({ id: s.id, name: s.name, loaded: s.id === activeSession, hasCloud: s.hasCloud }))}
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
                activeScanKey={activeScanTab} rebuildingSession={pipelineRunning?.status === 'running' ? pipelineRunning.session_id ?? null : null}
                extracting={extractingSessions} canManage={canManage}
                onConnect={connectToServer} onSelect={handleSessionSelect} onLoad={handleSessionLoad}
                onFlythrough={id => { if (activeSession !== id) handleSessionLoad(id); setFlythroughOpen(id) }}
                onReconstruct={handleReconstruct} onPickVideo={handlePickVideo} onSegment={handleSegment} onOpenManager={openSegmentationManager}
                onUnload={handleUnload} onRename={renameSession} onDelete={deleteSession} onCreate={createSession}
                onOpenScan={openScanFromTree} onSetReference={setReference} />
            )}
            {layout.state.leftTab === 'instances' && activeSession && (
              <InstancesPanel segments={segments} unsegmentedVisible={unsegmentedVisible} unsegmentedCount={unsegmentedCount} absorbedCount={absorbedCount} floorLevel={floorLevel}
                placedObjects={placedObjects} shapeMeshes={shapeMeshes} tsdfMeshes={tsdfMeshes} selectedSegmentId={selectedSegmentId}
                canFuse={projectScans.filter((sc: any) => sc.kind !== 'fused').length >= 2}
                onSelectSegment={id => {
                  setSelectedSegmentId(id)
                  if (id == null) return
                  layout.openInspector('properties')
                  // FLY TO the object (USER 2026-09-21). The frame comes from
                  // the instance's own OBB, so a 9 m floor and a 20 cm light
                  // both fill the view; an instance with no OBB (just
                  // propagated, no points) simply does not move the camera.
                  const seg = segments.find(s => s.id === id)
                  if (seg?.focus) viewportRef.current?.flyToPoint(seg.focus.center, seg.focus.radius)
                }}
                onSelectAll={() => {
                  setSegments(prev => prev.map(s => ({ ...s, visible: true }))); setUnsegmentedVisible(true)
                  segments.forEach(s => { viewportRef.current?.toggleOBB(s.key, true); viewportRef.current?.setSegmentVisibility(s.classId, true) })
                  viewportRef.current?.setSegmentVisibility(0, true)
                }}
                onDeselectAll={() => {
                  setSegments(prev => prev.map(s => ({ ...s, visible: false }))); setUnsegmentedVisible(false)
                  segments.forEach(s => { viewportRef.current?.toggleOBB(s.key, false); viewportRef.current?.setSegmentVisibility(s.classId, false) })
                  viewportRef.current?.setSegmentVisibility(0, false)
                }}
                onToggleSegment={(seg, vis) => { setSegments(prev => prev.map(s => (s.key === seg.key ? { ...s, visible: vis } : s))); viewportRef.current?.toggleOBB(seg.key, vis); viewportRef.current?.setSegmentVisibility(seg.classId, vis) }}
                onRenameSegment={renameSegment} onDeleteSegment={deleteSegment}
                onToggleUnsegmented={v => { setUnsegmentedVisible(v); viewportRef.current?.setSegmentVisibility(0, v) }}
                onFloorLevel={iid => applyFloorLevel(activeSession, 'explicit', iid)}
                onTogglePlaced={(id, v) => viewportRef.current?.setSceneObjectVisible(id, v)} onRemovePlaced={id => viewportRef.current?.removeSceneObject(id)}
                onToggleShape={(m, v) => { setShapeMeshes(prev => prev.map(x => (x.folder === m.folder ? { ...x, visible: v } : x))); viewportRef.current?.setShapeVisibility(m.instanceId, v) }}
                onToggleTsdf={(m, v) => { setTsdfMeshes(prev => prev.map(x => (x.folder === m.folder ? { ...x, visible: v } : x))); viewportRef.current?.setTsdfVisibility(m.folder, v) }}
                onDeleteTsdf={deleteTsdfMesh}
                onOpenSegmentation={() => openSegmentationManager(activeSession)} onOpenMeshing={openTsdfModal} onOpenCorrection={openCorrection} onOpenFuse={() => setShowFuseModal(true)} />
            )}
            {layout.state.leftTab === 'scans' && (
              <ScansPanel sessionId={activeSession} scans={projectScans} activeScanKey={activeScanTab}
                onOpenScan={sc => activeSession && openScanFromTree(activeSession, sc)} onSetReference={sc => activeSession && setReference(activeSession, sc)} onFuse={() => setShowFuseModal(true)} />
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
              const isActive = tab.key === activeScanTab
              return {
                id: tab.key, closable: true, title: `${tab.date} ${tab.label}${sc?.is_reference ? ` (${t('header.reference')})` : ''}`,
                icon: sc?.is_reference ? <Star className="stac-scantabs__ref" aria-hidden /> : tab.kind === 'fused' ? <Layers aria-hidden /> : undefined,
                label: tab.kind === 'fused' ? tab.label : `${tab.date} ${tab.label}`,
                badge: isActive && correctionState?.status === 'applied' ? <AlertTriangle className="stac-scantabs__warn" aria-label={t('correction.pendingShort')} /> : undefined,
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
            pipelineRunning={!!pipelineRunning && pipelineRunning.status === 'running'}
            onPointCount={n => { setPointCount(n); if (n > 0) lastCloudLoadAtRef.current = Date.now() }}
            onFps={setFps}
            onStatusMessage={setStatusMessage}
            onSegments={list => { setSegments(list); refreshUnsegmentedCount(activeSession) }}
            onEraseLedger={() => reloadSegments(activeSession)}
            onTsdfReady={sid => refreshTsdfMeshList(sid)}
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
            onHasCameraPoses={setHasCameraPoses}
            onCursor={setCursor}
            onReadouts={setReadouts}
            onViewScale={setMetersPerPixel}
          />

          {hasSession && !sessionLoading && (
            <FloatingToolbar
              activeTool={activeTool} onTool={setActiveTool} onResetCamera={() => viewportRef.current?.resetCamera()}
              onClearMeasurements={() => viewportRef.current?.clearMeasurements()} onAddObject={openObjectLibrary}
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
              }}
              display={{ pointSize, onPointSize: setPointSize, budgetM: pointBudget / 1_000_000, onBudgetM: v => setPointBudget(v * 1_000_000), hasConfidence, confidence: confidenceThreshold, onConfidence: setConfidenceThreshold }}
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

          {hasSession && !sessionLoading && (
            <ViewportHud onView={preset => viewportRef.current?.setStandardView(preset)} metersPerPixel={metersPerPixel}
              colorMode={sabanaVisible ? 'deviation' : colorMode} mvThreshold={mvThreshold}
              statusFractions={certifyState?.report_metrics?.witnesses?.status_fraction ?? null}
              deviationRange={sabanaVisible && sabanaMetrics ? { maxMm: (sabanaMetrics.tolerance_mm || 15) * 3, toleranceMm: sabanaMetrics.tolerance_mm || 15 } : null}
              showEdges={showCertifyKit} readouts={readouts}
              banner={pipelineActiveHere && pointCount > 0 && currentStage ? (
                <Banner glass compact tone="brand" title={t('pipeline.runningBanner', { stage: currentStage.label })} action={<Button size="sm" variant="ghost" onClick={handlePipelineCancel}>{t('common.cancel')}</Button>}>
                  {currentStage.message} {fmt.percent(currentStage.pct / 100)}
                </Banner>
              ) : undefined} />
          )}

          {flythroughOpen && <SyncPlayer sessionId={flythroughOpen} viewportRef={viewportRef} onClose={() => setFlythroughOpen(null)} />}

          {loadingOverlay && <SplashOverlay title={t('splash.loadingSession')} status={sessionLoading || undefined} />}
          {pipelineActiveHere && pointCount === 0 && (
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
              ]}
              panels={{
                console: <ConsolePanel logs={consoleLogs} />,
                jobs: <JobsPanel pipeline={pipelineRunning ? { session_id: pipelineRunning.session_id, status: pipelineRunning.status, stages: pipelineStages } : null} onCancelPipeline={handlePipelineCancel}
                  correction={correctionRunning ? correctionProgress : null} meshing={tsdfRunning ? tsdfOverall : null} object={shapeRunning ? shapeOverall : null}
                  resume={resumeBusy ? resumeProgress : null} extracting={extractingSessions} tasks={activeTasks} />,
                timeline: <ScansPanel variant="timeline" sessionId={activeSession} scans={projectScans} activeScanKey={activeScanTab}
                  onOpenScan={sc => activeSession && openScanFromTree(activeSession, sc)} onSetReference={sc => activeSession && setReference(activeSession, sc)} onFuse={() => setShowFuseModal(true)} />,
                report: activeSession && bimModels.length > 0 && segments.length > 0
                  ? <DeviationOverlay sessionId={activeSession} viewportRef={viewportRef} onClose={() => layout.setDockOpen(false)} />
                  : <EmptyState compact icon={<BarChart3 aria-hidden />} title={t('report.emptyTitle')} description={t('report.emptyDesc')} />,
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
              assistant: <AssistantPanel sessionId={activeSession} viewport={viewportRef} onVlmStatus={setVlmStatus} />,
            }} />
        )}
      </div>

      <div className="stac-app__status">
        <StatusBar serverAlive={serverAlive} connected={connected} sessionLabel={activeSession}
          scanLabel={activeScanRow ? `${activeScanRow.date} ${activeScanRow.label}` : null}
          message={statusMessage} epoch={epoch} storedEpochs={storedEpochs} cursor={cursor}
          job={pipelineActiveHere && currentStage ? currentStage.label : correctionRunning ? t('jobs.correction') : tsdfRunning ? t('jobs.meshing') : null}
          jobPct={pipelineActiveHere && currentStage ? currentStage.pct : correctionRunning ? correctionProgress?.pct ?? null : null}
          points={pointCount} fps={fps} consoleOpen={consoleOpen} onToggleConsole={() => setConsoleOpen(!consoleOpen)} />
      </div>

      <PipelineDialog open={pipelineDialogOpen} sessionId={pipelineDialogSession} scans={scansList} selected={selectedScans}
        onToggleScan={(key, on) => setSelectedScans(prev => (on ? [...prev, key] : prev.filter(k => k !== key)))}
        replace={pipelineReplace} onReplace={setPipelineReplace}
        rebuildingKeys={(() => { const pr = pipelineRunning; const active = !!(pr && (pr.status === 'running' || pr.status === 'queued') && pr.session_id === pipelineDialogSession); return active ? (pr!.scans || scansList.map(s => s.key)) : [] })()}
        onCancel={() => setPipelineDialogOpen(false)} onRun={handlePipelineRun} />

      {adminOpen && <AdminPage onClose={() => setAdminOpen(false)} />}

      {(callTarget || incomingCall) && user && (
        <WebRTCCall wsRef={teamWsRef} userId={user.id} callTarget={callTarget} incomingCall={incomingCall}
          onClose={() => { setCallTarget(null); setIncomingCall(null) }} onIncomingHandled={() => setIncomingCall(null)} />
      )}

      {showFuseModal && activeSession && (
        <FuseScansModal project={activeSession} scans={projectScans} onClose={() => setShowFuseModal(false)} onStatus={setStatusMessage}
          onFused={async () => {
            const scans = await loadProjectScans(activeSession)
            const fused = scans.filter(s => s.kind === 'fused').slice(-1)[0]
            if (fused) activateScan(activeSession, { key: fused.key, label: fused.label, date: fused.date, kind: 'fused' })
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
            if (d.ok) { setShowObjectLibrary(false); await viewportRef.current?.placeSceneObject(d.object) }
          } catch { report(t('library.addFailed'), 'err') }
        }} />

      <CorrectionDialog open={showCorrectionModal} onClose={() => setShowCorrectionModal(false)} state={correctionState} report={correctionReport} running={correctionRunning} progress={correctionProgress}
        segments={segments} selected={correctionSelected} onToggleSelected={id => setCorrectionSelected(prev => { const n = new Set(prev); if (n.has(id)) n.delete(id); else n.add(id); return n })}
        overrideScale={correctionOverrideScale} onOverrideScale={setCorrectionOverrideScale} ledger={correctionLedger} artifacts={correctionArtifacts}
        onRun={() => runCorrectionOp('objects')} onFloor={() => runCorrectionOp('floor')} onRevisit={() => runCorrectionOp('revisit')} epochs={correctionEpochs} onSelectEpoch={selectCorrectionEpoch} />

      <MeshingDialog open={showTsdfModal} onClose={() => setShowTsdfModal(false)} segments={segments} selected={tsdfSelected}
        onToggle={id => setTsdfSelected(prev => { const n = new Set(prev); if (n.has(id)) n.delete(id); else n.add(id); return n })}
        tsdfStatus={tsdfStatus} tsdfProgress={tsdfProgress} tsdfOverall={tsdfOverall} tsdfRunning={tsdfRunning} shapeRunning={shapeRunning} shapeOverall={shapeOverall}
        onObject={runObjectMeshing} onMesh={runMesh} />

      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} commands={commands} />
      <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      {appDialog}
    </div>
  )
}

export default App
