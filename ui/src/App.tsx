/**
 * STAC Build — Main Application
 * Hernán Barreto — Ingerop IN3
 */
import { useState, useCallback, useEffect, useRef } from 'react'
import './App.css'
import Viewport, { ViewportHandle, SegmentInstance } from './components/Viewport'
import { SyncPlayer } from './components/SyncPlayer'
import { BIMAnalysisPanel } from './components/BIMAnalysisPanel'
import TeamPanel from './components/TeamPanel'
import WebRTCCall from './components/WebRTCCall'
import BIMNavigator from './components/BIMNavigator'
import type { IFCLoadResult } from './components/IFCLoader'
import { useAuth } from './context/AuthContext'
import LoginPage from './pages/LoginPage'
import AdminPage from './pages/AdminPage'
import SegmentationManager from './components/InteractiveSegmentation'
import { useConfirmDialog } from './components/ConfirmDialog'
import {
  Search, Tag, Hammer, Plug, Upload, Settings, Crosshair,
  Maximize, Monitor, Grid3X3, RotateCcw, Ruler, TriangleRight, Scissors,
  Move, Palette, BookOpen, Keyboard, Info, Users, LogOut, FolderOpen, Axis3D,
  Building2, ArrowUpFromLine, ChevronLeft, ChevronRight, Trash2, Unlock, Play, X,
  Clock, CheckCircle2, XCircle, Ban, Circle, CheckSquare, Square, Check,
  Scale, Thermometer, Loader2, BarChart3, Home, Pencil, Camera, Plus, SlidersHorizontal,
  Sparkles, Eraser, Undo2, Brush,
} from 'lucide-react'
import AssistantPanel from './components/AssistantPanel'

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
  const { confirmDanger, dialogElement: appDialog } = useConfirmDialog()
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [activeSession, setActiveSession] = useState<string | null>(null)
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [activeTool, setActiveTool] = useState<Tool>('navigate')
  const [eraseRadius, setEraseRadius] = useState(0.15)
  const [eraseMarks, setEraseMarks] = useState(0)
  const [eraseTarget, setEraseTarget] = useState<string>('')
  const [eraseShape, setEraseShape] = useState<'sphere' | 'cube'>('sphere')
  const [eraseYawDeg, setEraseYawDeg] = useState(0)
  const [connected, setConnected] = useState(false)
  const [serverAlive, setServerAlive] = useState(false)
  const [activePanel, setActivePanel] = useState<'sessions' | 'segments' | 'bim' | 'team' | 'analysis' | 'assistant' | null>('sessions')

  const [bimModels, setBimModels] = useState<IFCLoadResult[]>([])
  const [sidebarWidth, setSidebarWidth] = useState(280)
  // Right-side collapsible AI chat dock (user 2026-08-28: the chat lives on the
  // right, always at hand — works with or without a session/segmentation).
  const [assistantOpen, setAssistantOpen] = useState<boolean>(
    () => localStorage.getItem('stac.assistantOpen') !== '0')
  const [assistantWidth, setAssistantWidth] = useState<number>(
    () => Number(localStorage.getItem('stac.assistantWidth')) || 360)
  // Model state reported by the chat panel — drives the menubar icon (gray
  // until the model is loaded, lit when up).
  const [vlmStatus, setVlmStatus] = useState<'up' | 'loading' | 'busy' | 'down' | null>(null)
  const toggleAssistant = useCallback(() => {
    setAssistantOpen((o) => {
      localStorage.setItem('stac.assistantOpen', o ? '0' : '1')
      return !o
    })
  }, [])
  const [consoleOpen, setConsoleOpen] = useState(false)
  const [pointSize, setPointSize] = useState(5.0)
  // Potree LOD point budget (max points rendered at once). Higher = more of the
  // cloud visible, but more client GPU/RAM. Slider lets it scale to any
  // cloud/machine. Default 10M (good overview for large clouds; raise if your
  // viewer GPU handles it).
  const [pointBudget, setPointBudget] = useState(10_000_000)
  // Display-settings popover (groups Point Size / Detail / Confidence so they
  // don't eat toolbar space).
  const [showDisplayMenu, setShowDisplayMenu] = useState(false)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.0)
  const [hasConfidence, setHasConfidence] = useState(false)

  const [showAxes, setShowAxes] = useState(false)
  const [showGrid, setShowGrid] = useState(true)
  const [pointCount, setPointCount] = useState(0)
  const [sessionLoading, setSessionLoading] = useState<string | null>(null)
  const [fps, setFps] = useState(0)
  const [consoleLogs, setConsoleLogs] = useState<{ ts: string; level: string; msg: string }[]>([])
  const consoleEndRef = useRef<HTMLDivElement>(null)
  const [openMenu, setOpenMenu] = useState<string | null>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const viewportRef = useRef<ViewportHandle>(null)
  // Timestamp (ms) of the last time a cloud actually loaded into the viewer (point
  // count > 0). Used by the pipeline-done handler to tell "the cloud arrived on this
  // live socket" from "WS reconnected mid-pipeline → potree_ready went to a dead socket".
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

  // The pipeline always runs end-to-end (reconstruction → cloud cleaning →
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
  const [creatingProject, setCreatingProject] = useState(false)
  const [newProjectName, setNewProjectName] = useState('')
  const [renamingProject, setRenamingProject] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  // Video upload → frame extraction, keyed by session id. Presence means the
  // session has no frames yet and is uploading/extracting; pct drives the spinner.
  const [extractingSessions, setExtractingSessions] = useState<Record<string, number>>({})
  const videoInputRef = useRef<HTMLInputElement>(null)
  const pendingVideoSession = useRef<string | null>(null)
  const [scansList, setScansList] = useState<{ date: string; source: string; key: string; frame_count: number; has_output: boolean; recon_state?: string; cached_chunks?: number }[]>([])
  const [selectedScans, setSelectedScans] = useState<string[]>([])
  const [projectFilter, setProjectFilter] = useState('')

  const { user, token, loading: authLoading, logout } = useAuth()



  // Periodic server health check — updates indicator only
  // NOTE: timeout/threshold are generous because the Python server does heavy
  // CPU-bound work (BIM registration, SAM3 loading, reconstruction) that blocks
  // the GIL for up to several minutes. Short thresholds cause false disconnects.
  const failCountRef = useRef(0)
  const FAIL_THRESHOLD = 10 // ~60s of unresponsiveness before declaring dead
  useEffect(() => {
    interactiveSessionRef.current = interactiveSessionId
  }, [interactiveSessionId])
  useEffect(() => {
    let intervalMs = 5000
    let timerId: ReturnType<typeof setTimeout> | null = null
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
          intervalMs = 5000
        }
      } catch {
        failCountRef.current++
        if (failCountRef.current >= FAIL_THRESHOLD) setServerAlive(false)
        intervalMs = failCountRef.current >= FAIL_THRESHOLD ? 10000 : 5000
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
      // Server just went from alive → dead (after 3 consecutive failures)
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
      setStatusMessage('Server disconnected')
      // Dispose Potree data from GPU to free memory and stop LOD updates
      viewportRef.current?.clearScene()
    }
    if (!prevAliveRef.current && serverAlive) {
      // Server just went from dead → alive: clear stale message
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

  // Auto-scroll console
  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [consoleLogs])

  const [statusMessage, setStatusMessage] = useState('')
  const [segments, setSegments] = useState<SegmentInstance[]>([])
  // Floor→y=0 leveling: candidates + which floor instance sits at y=0.
  // Changing the combobox re-levels instantly (no confirm); the manager
  // close and session load call the auto modes.
  const [floorLevel, setFloorLevel] = useState<{ candidates: { instance_id: number, label: string, height_m: number | null }[], selected: number | null }>({ candidates: [], selected: null })
  const floorLevelCheckedRef = useRef<string | null>(null)
  const [editingSegKey, setEditingSegKey] = useState<string | null>(null)
  const [segSearch, setSegSearch] = useState('')

  // Cloud object-level visibility — App owns the truth (Unsegmented + segments).
  // Reset to "visible" on every session switch; the second useEffect combines
  // it with `segments` and tells the Viewport whether to keep the cloud in the
  // scene (true) or pull it out (false → frees GPU, the Potree LOD self-gates,
  // and the raycaster skips it so measurements land on meshes).
  const [unsegmentedVisible, setUnsegmentedVisible] = useState(true)
  useEffect(() => { setUnsegmentedVisible(true) }, [activeSession])
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

  // Shape export state
  const [showShapeModal, setShowShapeModal] = useState(false)
  const [shapeAutoReconstruct, setShapeAutoReconstruct] = useState(true)
  const [shapeRunning, setShapeRunning] = useState(false)
  // Ref-based busy guard. The `disabled` prop on the Start button only gates
  // clicks AFTER React commits the next render — between the click and that
  // commit (a few ms in the worst case, more under load) the DOM button is
  // still clickable. A bouncy mouse, an over-eager screen reader, or an
  // intermediary retry can all sneak a second click through. The ref is set
  // synchronously, so any second entry within the same JS turn is rejected.
  const shapeBusyRef = useRef(false)
  // Reconstruction-v2 ("scene") trigger — assembles parametric surfaces / swept
  // solids / boxes / linear-repeats + free-form generated (MeshFlow) meshes into output/scene/.
  const [reconRunning, setReconRunning] = useState(false)
  const reconBusyRef = useRef(false)
  const [shapeResult, setShapeResult] = useState<{ count: number; exported: string[] } | null>(null)
  const [shapeSelected, setShapeSelected] = useState<Set<number>>(new Set())
  const [shapeStatus, setShapeStatus] = useState<Record<number, { has_pkl: boolean; has_mesh: boolean }>>({})

  // Live progress for the in-flight shape run
  type ShapeInstanceProgress = {
    id: number
    phase: string  // captioning | exporting_pkl | pkl_ready | reconstructing | done | error
    elapsed?: number
    error?: string
    mesh?: string
  }
  const [shapeProgress, setShapeProgress] = useState<Record<number, ShapeInstanceProgress>>({})
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
            || (s.meta?.method === 'tsdf_scene' ? '🧱 TSDF — whole scene'
                : s.meta?.method === 'poisson_scene' ? '🟣 Poisson — whole scene'
                : s.folder === 'scene' ? '🌐 Whole scene' : s.folder),
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

  // (the old standalone Shape modal opener was removed — the 🧩 Meshing modal's
  // Object button covers it, sharing the same segment selection)

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

  // Poll reconstruction-v2 progress while a run is in flight. The POST kicks
  // off a background job (single-flight, so it can't OOM the host) and returns
  // immediately; this drives the status line and reloads the scene when done.
  useEffect(() => {
    if (!reconRunning || !activeSession) return
    let alive = true
    const tick = async () => {
      try {
        const r = await fetch(`/api/segmentation/reconstruct/progress/${activeSession}`)
        const p = await r.json()
        if (!alive) return
        const ph: string = p.phase || 'running'
        if (ph === 'done') {
          const s = p.summary || {}
          const byClass = Object.entries(s.by_class || {}).map(([k, v]) => `${v} ${k}`).join(', ')
          setStatusMessage(`Scene: ${s.n_elements ?? '?'} element(s) [${byClass}], ${s.n_adjacency ?? 0} adjacency edge(s) — ${s.elapsed_s ?? '?'}s`)
          try { await viewportRef.current?.reloadReconScene(activeSession) } catch { /* ignore */ }
          reconBusyRef.current = false
          setReconRunning(false)
          return
        }
        if (ph === 'error') {
          setStatusMessage(`Reconstruction error: ${p.error || 'unknown'}`)
          reconBusyRef.current = false
          setReconRunning(false)
          return
        }
        const det = (ph === 'classifying' && p.total)
          ? ` ${p.done ?? 0}/${p.total}${p.current ? ' — ' + p.current : ''}` : ''
        const nv = p.n_views != null ? ` · ${p.n_views} views` : ''
        const npts = p.n_points != null ? ` · ${(p.n_points / 1e6).toFixed(1)}M pts` : ''
        setStatusMessage(`Reconstruction: ${ph}${det}${nv}${npts}…`)
      } catch { /* keep polling through transient errors */ }
    }
    tick()
    const id = window.setInterval(tick, 1500)
    return () => { alive = false; window.clearInterval(id) }
  }, [reconRunning, activeSession])

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

  // Close menu when clicking outside
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpenMenu(null)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Home') { viewportRef.current?.resetCamera(); e.preventDefault() }
      if (e.key === '`' && e.ctrlKey) {
        setConsoleOpen(prev => !prev)
        e.preventDefault()
      }
      if (e.key === 'b' && e.ctrlKey) {
        setActivePanel(prev => prev ? null : 'sessions')
        e.preventDefault()
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [])

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
      // console.log(`[STAC] Connected. ${sessionList.length} sessions found.`)
    } catch (err) {
      console.error('[STAC] Failed to connect:', err)
      setConnected(false)
      setActivePanel('sessions')
    }
  }, [token])

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
  }, [sessions]) // eslint-disable-line react-hooks/exhaustive-deps

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
          setStatusMessage('Session has no point cloud data. Run Reconstruct to generate.')
        }
      }
    }, 45000)  // 45s: large octrees need time to download before points appear
    return () => clearTimeout(timer)
  }, [sessionLoading, pipelineRunning, activeSession]) // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-detect sábana when BIM models first appear after session load
  const prevBimCount = useRef(0)
  useEffect(() => {
    const wasZero = prevBimCount.current === 0
    prevBimCount.current = bimModels.length
    // Only trigger when transitioning from 0 → N models (fresh session load)
    if (!wasZero || bimModels.length === 0 || !activeSession) return
    fetch(`/api/sessions/${activeSession}/sabana/exists`)
      .then(r => r.json())
      .then(d => {
        if (d.exists) {
          setSabanaVisible(true)
          setSabanaLoading(true)
          setSessionLoading('Loading sábana comparison...')
          setStatusMessage('Loading sábana comparison...')
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
              const segRes = await fetch(`/api/sessions/${activeSession}/segmentation`)
              if (segRes.ok) {
                const segData = await segRes.json()
                if (Array.isArray(segData.instances)) {
                  setSegments(segData.instances.map((inst: any) => ({
                    key: inst.global_id || `${inst.label}_${inst.instance_id || inst.id}`,
                    id: inst.instance_id || inst.id,
                    label: `${inst.label}`,
                    color: inst.color || '#4fd1ff',
                    totalPoints: inst.total_points || 0,
                    visible: true,
                    excluded: inst.excluded || false,
                  })))
                }
              }
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
  }, [activeSession, connected]) // eslint-disable-line react-hooks/exhaustive-deps

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
          setStatusMessage(`Extracting frames: ${p.saved || 0}${p.total ? `/${p.total}` : ''} (${p.pct || 0}%)…`)
          setTimeout(tick, 1500)
        } else if (p.phase === 'error') {
          setExtractingSessions(prev => { const n = { ...prev }; delete n[sessionId]; return n })
          setStatusMessage(`Frame extraction failed: ${p.error || 'unknown'}`)
        } else {
          // done or idle → frames are on disk
          setExtractingSessions(prev => { const n = { ...prev }; delete n[sessionId]; return n })
          setStatusMessage(`Frames extracted (${p.saved || p.frame_count || 0}). Ready to reconstruct.`)
          connectToServer()
        }
      } catch {
        setExtractingSessions(prev => { const n = { ...prev }; delete n[sessionId]; return n })
        setStatusMessage('Lost connection while extracting frames')
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
    setStatusMessage(`Uploading ${file.name}…`)
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
        setStatusMessage(`Upload failed: ${err.detail || `HTTP ${res.status}`}`)
        return
      }
      pollVideoExtraction(sessionId)
    } catch (err: any) {
      setExtractingSessions(prev => { const n = { ...prev }; delete n[sessionId]; return n })
      setStatusMessage(`Upload error: ${err?.message ?? err}`)
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
        setStatusMessage(`Floor leveled to y=0 (was ${d.residual_before_mm ?? '?'} mm off)`)
      }
    } catch { /* non-fatal */ }
  }, [])

  // Floor→y=0 on session load: once the cloud is actually in the viewer
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
      const label = selectedScans.length === 1 ? selectedScans[0] : `${selectedScans.length} scans`
      setStatusMessage(`Pipeline started for ${pipelineDialogSession} (${label})...`)
    }, 500)
  }, [pipelineDialogSession, pipelineReplace, selectedScans])

  const handlePipelineCancel = useCallback(() => {
    const targetSession = pipelineRunning?.session_id || activeSession
    if (!targetSession) return
    viewportRef.current?.sendCommand({ type: 'cancel_pipeline', session_id: targetSession })
    setPipelineRunning(null)
    setStatusMessage('Pipeline cancelled')
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
      setStatusMessage(`${currentStage.icon} ${currentStage.label}: ${shortMsg || ''} ${pct}%`)
    }
    // Dead-socket FALLBACK only: if the WS reconnected during a long pipeline, the
    // backend's _on_pipeline_complete closure sent potree_ready on the now-dead socket,
    // so no cloud arrives on THIS live socket → reload the session. If the cloud DID load
    // here (the normal case), do NOTHING. The old code unconditionally clearScene()'d +
    // reloaded, which WIPED the cloud potree_ready had just loaded → the viewer got stuck
    // and only a backend restart + manual reload recovered it. No clearScene now: the
    // potree_ready handler already disposes/replaces the loader on reload.
    if (newState.status === 'done' && newState.session_id) {
      const sid = newState.session_id
      const doneAt = Date.now()
      setTimeout(() => {
        if (lastCloudLoadAtRef.current >= doneAt) return  // cloud arrived on this socket → keep it
        viewportRef.current?.sendCommand({ type: 'load_session', session_id: sid })
      }, 8000)
    }
  }, [])

  const handleSegment = useCallback((sessionId: string) => {
    viewportRef.current?.sendCommand({ type: 'set_prompt', prompt: 'auto' })
    setStatusMessage(`Segmenting ${sessionId}...`)
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
    if (!wasZero || pointCount === 0) return  // Only trigger on 0 → N transition
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
  }, [pointCount]) // eslint-disable-line react-hooks/exhaustive-deps

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
  }, [pointSize, confidenceThreshold, pointBudget]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Sábana: Generate comparison (auto_match → compare → show via Potree) ──
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
    setStatusMessage(useManualAlignment ? 'Running BIM comparison (manual alignment)...' : 'Running BIM comparison (auto-register)...')
    try {
      const matchRes = await fetch(`/api/bim/auto_match/${activeSession}`)
      const matchData = await matchRes.json()
      if (!matchData.matches?.length) {
        setStatusMessage('No matches found between segments and IFC elements')
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
        setStatusMessage(`Comparison failed: ${compareData.error || 'unknown'}`)
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
      setStatusMessage(`Comparison error: ${e.message}`)
    }
    setSabanaLoading(false)
  }, [activeSession, sabanaVisible, useManualAlignment])

  // ── Sábana: Toggle visibility via Potree streaming ──
  const handleToggleSabana = useCallback(() => {
    if (!activeSession) return
    if (sabanaVisible) {
      // Toggle OFF → reload original scan cloud + show OBBs (preserve camera!)
      viewportRef.current?.sendCommandPreserveCamera({ type: 'load_session', session_id: activeSession })
      viewportRef.current?.setOBBsVisible(true)
      setSabanaVisible(false)
      setSabanaMetrics(null)
      setSabanaFullMeta(null)
      setStatusMessage('Reloading scan cloud...')
      setActivePanel('bim')
      return
    }
    // Toggle ON → load sábana + hide OBBs + fetch analysis meta
    setStatusMessage('Loading sábana...')
    viewportRef.current?.sendCommand({ type: 'load_sabana', session_id: activeSession })
    viewportRef.current?.setOBBsVisible(false)
    if (activeTool === 'align') setActiveTool('navigate')
    setSabanaVisible(true)
    // Fetch full metadata → show analysis panel
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

  const toggleMenu = (menu: string) => {
    setOpenMenu(openMenu === menu ? null : menu)
  }

  const menuAction = (action: () => void) => {
    action()
    setOpenMenu(null)
  }

  const hasSession = activeSession !== null
  const panelOpen = activePanel !== null

  const togglePanel = (panel: typeof activePanel) => {
    setActivePanel(prev => prev === panel ? null : panel)
  }

  if (authLoading) {
    return (
      <div className="login-page">
        <div className="login-card">
          <div className="login-logo spinning">S</div>
          <p className="login-subtitle">Loading…</p>
        </div>
      </div>
    )
  }

  if (!user) return <LoginPage />

  return (
    <div
      className={`app-layout ${!panelOpen ? 'panel-collapsed' : ''}`}
      style={{
        ...(panelOpen ? { '--sidebar-width': `${sidebarWidth}px` } : {}),
        '--assistant-width': assistantOpen ? `${assistantWidth}px` : '0px',
      } as React.CSSProperties}
    >
      {/* Hidden input used by the per-session "+" button to upload a video */}
      <input
        ref={videoInputRef}
        type="file"
        accept="video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,.mp4,.avi,.mov,.mkv,.m4v"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) handleVideoSelected(f)
        }}
      />
      {/* ── Menu Bar ── */}
      <div className="menubar" ref={menuRef}>
        <span className="menu-app-title"><img src="/favicon.ico" alt="" className="menu-app-logo" /> STAC Build</span>

        {/* File Menu */}
        <div className="menu-item">
          <button className={`menu-trigger ${openMenu === 'file' ? 'open' : ''}`}
            onClick={() => toggleMenu('file')}>File</button>
          {openMenu === 'file' && (
            <div className="menu-dropdown">
              {!connected ? (
                <button className="menu-dropdown-item"
                  onClick={() => menuAction(connectToServer)}>
                  <Plug size={14} /> Connect to Server
                </button>
              ) : (
                <button className="menu-dropdown-item"
                  onClick={() => menuAction(() => {
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
                  })}>
                  <ArrowUpFromLine size={14} /> Disconnect
                </button>
              )}
              <div className="menu-separator" />
              <button className="menu-dropdown-item" disabled={!hasSession}>
                <Upload size={14} /> Export Point Cloud
                <span className="menu-shortcut">Ctrl+E</span>
              </button>
              <div className="menu-separator" />
              <button className="menu-dropdown-item" disabled>
                <Settings size={14} /> Settings
                <span className="menu-shortcut">Ctrl+,</span>
              </button>
            </div>
          )}
        </div>

        {/* View Menu */}
        <div className="menu-item">
          <button className={`menu-trigger ${openMenu === 'view' ? 'open' : ''}`}
            onClick={() => toggleMenu('view')}>View</button>
          {openMenu === 'view' && (
            <div className="menu-dropdown">
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => togglePanel('sessions'))}>
                {panelOpen ? <><ChevronLeft size={14} /> Hide Panel</> : <><ChevronRight size={14} /> Show Panel</>}
                <span className="menu-shortcut">Ctrl+B</span>
              </button>
              <div className="menu-separator" />
              <button className="menu-dropdown-item" disabled={!hasSession}
                onClick={() => menuAction(() => viewportRef.current?.resetCamera())}>
                <Home size={14} /> Reset Camera
                <span className="menu-shortcut">Home</span>
              </button>
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => {
                  if (document.fullscreenElement) document.exitFullscreen()
                  else document.documentElement.requestFullscreen()
                })}>
                <Maximize size={14} /> Fullscreen
                <span className="menu-shortcut">F11</span>
              </button>
              <div className="menu-separator" />
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => {
                  setConsoleOpen(prev => !prev)
                })}>
                {consoleOpen ? <><Monitor size={14} /> Console <Check size={12} /></> : <><Monitor size={14} /> Console</>}
                <span className="menu-shortcut">Ctrl+`</span>
              </button>
              <div className="menu-separator" />
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => setShowAxes(prev => !prev))}>
                {showAxes ? <><Axis3D size={14} /> Axes <Check size={12} /></> : <><Axis3D size={14} /> Axes</>}
              </button>
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => setShowGrid(prev => !prev))}>
                {showGrid ? <><Grid3X3 size={14} /> Grid <Check size={12} /></> : <><Grid3X3 size={14} /> Grid</>}
              </button>
            </div>
          )}
        </div>

        {/* Tools Menu */}
        <div className="menu-item">
          <button className={`menu-trigger ${openMenu === 'tools' ? 'open' : ''}`}
            onClick={() => toggleMenu('tools')} disabled={!hasSession}>Tools</button>
          {openMenu === 'tools' && hasSession && (
            <div className="menu-dropdown">
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => setActiveTool('navigate'))}>
                <RotateCcw size={14} /> Navigate
                <span className="menu-shortcut">V</span>
              </button>
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => setActiveTool('measure-distance'))}>
                <Ruler size={14} /> Measure Distance
                <span className="menu-shortcut">M</span>
              </button>
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => setActiveTool('measure-angle'))}>
                <TriangleRight size={14} /> Measure Angle
                <span className="menu-shortcut">A</span>
              </button>
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => setActiveTool('section-box'))}>
                <Scissors size={14} /> Section Box
                <span className="menu-shortcut">X</span>
              </button>
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => setActiveTool(activeTool === 'erase' ? 'navigate' : 'erase'))}>
                <Eraser size={14} /> Erase Points
                {activeTool === 'erase' && <span className="menu-shortcut">ON</span>}
              </button>
              <button className="menu-dropdown-item"
                onClick={() => menuAction(async () => {
                  if (!activeSession) return
                  try {
                    const r = await fetch('/api/segmentation/erase/undo', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ session_id: activeSession }),
                    })
                    const d = await r.json()
                    setStatusMessage(d.ok ? `↩ erase undone (${(d.restored || 0).toLocaleString()} pts restored)` : '↩ nothing to undo')
                  } catch { setStatusMessage('↩ undo failed') }
                })}>
                <Undo2 size={14} /> Undo Erase
              </button>
              <button className="menu-dropdown-item"
                disabled={sabanaVisible}
                style={sabanaVisible ? { opacity: 0.4, pointerEvents: 'none' } : {}}
                onClick={() => menuAction(() => setActiveTool(activeTool === 'align' ? 'navigate' : 'align'))}>
                <Move size={14} /> Align Cloud
                <span className="menu-shortcut">G</span>
              </button>
              <div className="menu-separator" />
              <div className="menu-dropdown-item" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Palette size={14} /> Point Size
                <input type="range" min="0.01" max="5" step="0.01" value={pointSize}
                  onChange={e => setPointSize(parseFloat(e.target.value))}
                  onClick={e => e.stopPropagation()}
                  style={{ width: 80, accentColor: 'var(--accent)' }} />
                <span style={{ fontSize: 11, color: 'var(--text-secondary)', minWidth: 24 }}>{pointSize.toFixed(2)}</span>
              </div>
            </div>
          )}
        </div>

        {/* Help Menu */}
        <div className="menu-item">
          <button className={`menu-trigger ${openMenu === 'help' ? 'open' : ''}`}
            onClick={() => toggleMenu('help')}>Help</button>
          {openMenu === 'help' && (
            <div className="menu-dropdown">
              <button className="menu-dropdown-item" disabled>
                <BookOpen size={14} /> Documentation
              </button>
              <button className="menu-dropdown-item" disabled>
                <Keyboard size={14} /> Keyboard Shortcuts
                <span className="menu-shortcut">Ctrl+/</span>
              </button>
              <div className="menu-separator" />
              <button className="menu-dropdown-item" disabled>
                <Info size={14} /> About STAC Build
              </button>
            </div>
          )}
        </div>

        {/* User Menu — right side */}
        <div className="menu-spacer" />
        {/* AI chat dock toggle — top right, left of the logged-in user.
            Gray while the model is unloaded/loading; lit when it is up. */}
        <div className="menu-item">
          <button
            className={`menu-trigger chat-trigger ${assistantOpen ? 'open' : ''} ${vlmStatus === 'up' ? 'model-up' : 'model-off'}`}
            onClick={toggleAssistant}
            title={`AI Assistant — ${
              vlmStatus === 'up' ? 'model loaded'
              : vlmStatus === 'loading' ? 'loading the model…'
              : vlmStatus === 'busy' ? 'GPU busy — model unloaded'
              : 'model unloaded'}`}>
            <Sparkles size={14} />
          </button>
        </div>
        <div className="menu-item">
          <button className={`menu-trigger user-trigger ${openMenu === 'user' ? 'open' : ''}`}
            onClick={() => toggleMenu('user')}>
            <span className="user-avatar-small">{user.username[0].toUpperCase()}</span>
            {user.username}
          </button>
          {openMenu === 'user' && (
            <div className="menu-dropdown menu-dropdown-right">
              <div className="menu-dropdown-header">
                <strong>{user.full_name || user.username}</strong>
                <div className="menu-dropdown-role">{user.role}</div>
              </div>
              <div className="menu-separator" />
              {user.role === 'admin' && (
                <button className="menu-dropdown-item"
                  onClick={() => menuAction(() => setAdminOpen(true))}>
                  <Users size={14} /> User Management
                </button>
              )}
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => {
                  viewportRef.current?.sendCommand({ type: 'cleared' })
                  setConnected(false)
                  setActivePanel('sessions')
                  setSessions([])
                  setActiveSession(null)
                  setSelectedSession(null)
                  setSegments([])
                  setBimModels([])
                  setPointCount(0)
                  setActiveTool('navigate')
                  setStatusMessage('')
                  logout()
                })}>
                <LogOut size={14} /> Logout
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ── Activity Bar ── */}
      <div className="activity-bar">
        <button className={`activity-btn ${activePanel === 'sessions' ? 'active' : ''}`}
          onClick={() => togglePanel('sessions')} title="Projects">
          <FolderOpen size={18} />
          {sessions.length > 0 && <span className="activity-badge">{sessions.length}</span>}
        </button>

        {hasSession && (
          <button className={`activity-btn ${activePanel === 'segments' ? 'active' : ''}`}
            onClick={() => togglePanel('segments')} title="Segments">
            <Tag size={18} />
            {segments.length > 0 && <span className="activity-badge">{segments.length}</span>}
          </button>
        )}
        {hasSession && (
          <button className={`activity-btn ${activePanel === 'bim' ? 'active' : ''}`}
            onClick={() => togglePanel('bim')} title="BIM Navigator">
            <Building2 size={18} />
            {bimModels.length > 0 && <span className="activity-badge">{bimModels.length}</span>}
          </button>
        )}
        {hasSession && sabanaFullMeta && (
          <button className={`activity-btn ${activePanel === 'analysis' ? 'active' : ''}`}
            onClick={() => togglePanel('analysis')} title="BIM Analysis">
            <BarChart3 size={18} />
          </button>
        )}
        {/* Chat lives ONLY on the right dock (user 2026-08-28) — no toggle here */}
        <div className="activity-spacer" />
        <button className={`activity-btn ${activePanel === 'team' ? 'active' : ''}`}
          onClick={() => togglePanel('team')} title="Team">
          <Users size={18} />
        </button>
        <button className={`activity-btn ${consoleOpen ? 'active' : ''}`}
          onClick={() => setConsoleOpen(prev => !prev)} title="Console">
          <Monitor size={18} />
        </button>
        <button className="activity-btn" disabled title="Settings">
          <Settings size={18} />
        </button>
      </div>

      {/* ── Panel ── */}
      {panelOpen && (
        <div className="panel-column">
          <div className="panel">
            {/* Sessions Panel */}
            {activePanel === 'sessions' && (
              <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
                <div className="panel-header">Projects</div>
                <nav className="sidebar-nav" style={{ flex: 1, overflowY: 'auto' }}>
                  {!connected ? (
                    <>
                      <div className="nav-section">Server</div>
                      <div className="nav-item" onClick={connectToServer}>
                        <span className="nav-item-icon"><Plug size={14} /></span>
                        <span className="nav-item-label">Connect to Server</span>
                      </div>
                    </>
                  ) : (
                    <div className="session-list">
                      {/* Search filter — all users */}
                      <div className="session-search-bar">
                        <Search size={14} className="session-search-icon" />
                        <input
                          className="session-search-input"
                          placeholder="Search projects..."
                          value={projectFilter}
                          onChange={e => setProjectFilter(e.target.value)}
                        />
                        {projectFilter && (
                          <button className="session-search-clear" onClick={() => setProjectFilter('')}>✕</button>
                        )}
                      </div>
                      {sessions.filter(s => !projectFilter || s.name.toLowerCase().includes(projectFilter.toLowerCase())).map(s => (
                        <div
                          key={s.id}
                          className={`session-item ${selectedSession === s.id ? 'active' : ''} ${activeSession === s.id ? 'loaded' : ''}`}
                          onClick={() => handleSessionSelect(s.id)}
                        >
                          <div className="session-header">
                            <div className={`session-dot ${s.hasCloud ? 'online' : ''} ${activeSession === s.id ? 'loaded' : ''}`} />
                            {renamingProject === s.id ? (
                              <input
                                className="session-rename-input"
                                autoFocus
                                value={renameValue}
                                onClick={e => e.stopPropagation()}
                                onChange={e => setRenameValue(e.target.value)}
                                onKeyDown={async e => {
                                  if (e.key === 'Escape') { setRenamingProject(null); setRenameValue('') }
                                  if (e.key === 'Enter' && renameValue.trim() && renameValue.trim() !== s.id) {
                                    try {
                                      const headers: HeadersInit = { 'Content-Type': 'application/json' }
                                      if (token) headers['Authorization'] = `Bearer ${token}`
                                      const res = await fetch(`/sessions/${s.id}`, {
                                        method: 'PATCH', headers,
                                        body: JSON.stringify({ name: renameValue.trim() }),
                                      })
                                      if (!res.ok) {
                                        const err = await res.json().catch(() => ({}))
                                        setStatusMessage(err.detail || 'Rename failed')
                                      } else {
                                        setStatusMessage(`Renamed to "${renameValue.trim()}"`)
                                        connectToServer()
                                      }
                                    } catch { setStatusMessage('Rename failed') }
                                    setRenamingProject(null); setRenameValue('')
                                  }
                                }}
                                onBlur={() => { setRenamingProject(null); setRenameValue('') }}
                              />
                            ) : (
                              <div className="session-name">{s.name}</div>
                            )}
                          </div>
                          <div className="session-meta">
                            {s.frameCount} frames{s.hasCloud && ` · ${s.cloudSizeMb}MB`}
                            {s.hasSegments && <> · <Tag size={11} /></>}
                            {s.hasBim && <> · <Building2 size={11} />{s.bimCount > 1 ? ` (${s.bimCount})` : ''}</>}
                            {activeSession === s.id && <> · <Circle size={8} fill="var(--accent)" stroke="none" /> loaded</>}
                            {pipelineRunning && pipelineRunning.status === 'running' && pipelineRunning.session_id === s.id && (
                              <> · <span className="sidebar-pipeline-badge" title="Pipeline running">⚙️ rebuilding</span></>
                            )}
                          </div>
                          <div className="session-actions">
                            <button className="session-action-btn load"
                              title={s.hasCloud ? 'Load Session' : 'No cloud. Run Reconstruct first.'}
                              disabled={!s.hasCloud}
                              style={!s.hasCloud ? { opacity: 0.3 } : undefined}
                              onClick={(e) => { e.stopPropagation(); handleSessionLoad(s.id) }}
                            ><FolderOpen size={14} /></button>
                            <button className="session-action-btn flythrough"
                              title={s.hasCloud ? 'Flythrough: video ↔ escena 3D' : 'Sin nube todavía'}
                              disabled={!s.hasCloud}
                              style={!s.hasCloud ? { opacity: 0.3 } : undefined}
                              onClick={(e) => {
                                e.stopPropagation()
                                // use the already-loaded cloud/octree — only load if
                                // this session isn't the active one (avoid a reload).
                                if (activeSession !== s.id) handleSessionLoad(s.id)
                                setFlythroughOpen(s.id)
                              }}
                            ><Play size={14} /></button>
                            {s.id in extractingSessions ? (
                              <button className="session-action-btn reconstruct"
                                title={`Extracting frames… ${extractingSessions[s.id] || 0}%`}
                                disabled
                                onClick={(e) => e.stopPropagation()}
                              ><Loader2 size={14} className="spin" /></button>
                            ) : s.frameCount > 0 ? (
                              <button className="session-action-btn reconstruct"
                                title="Reconstruct Geometry"
                                onClick={(e) => { e.stopPropagation(); handleReconstruct(s.id) }}
                              ><Hammer size={14} /></button>
                            ) : (
                              (user?.role === 'admin' || user?.role === 'manager') && (
                                <button className="session-action-btn reconstruct"
                                  title="No frames yet — upload a video to extract frames"
                                  onClick={(e) => { e.stopPropagation(); handlePickVideo(s.id) }}
                                ><Plus size={14} /></button>
                              )
                            )}
                            {activeSession === s.id && (
                              <button className="session-action-btn segment"
                                title="Segment Objects"
                                onClick={(e) => { e.stopPropagation(); handleSegment(s.id) }}
                              ><Tag size={14} /></button>
                            )}
                            {activeSession === s.id && (
                              <button className="session-action-btn segment"
                                title="Manual Interactive Segmentation"
                                onClick={(e) => { e.stopPropagation(); setInteractiveSessionId(s.id) }}
                              ><Crosshair size={14} /></button>
                            )}
                            {activeSession === s.id && (
                              <button className="session-action-btn unload"
                                title="Unload Session"
                                onClick={(e) => { e.stopPropagation(); handleUnload() }}
                              ><ArrowUpFromLine size={14} /></button>
                            )}
                            {(user?.role === 'admin' || user?.role === 'manager') && (
                              <button className="session-action-btn reconstruct"
                                title="Rename Project"
                                onClick={(e) => { e.stopPropagation(); setRenamingProject(s.id); setRenameValue(s.id) }}
                              ><Pencil size={14} /></button>
                            )}
                            {(user?.role === 'admin' || user?.role === 'manager') && (
                              <button className="session-action-btn delete"
                                title="Delete Project"
                                onClick={async (e) => {
                                  e.stopPropagation()
                                  const ok = await confirmDanger(
                                    `This will permanently delete "${s.id}" and all its data. This cannot be undone.`,
                                    `Delete ${s.id}?`
                                  )
                                  if (!ok) return
                                  try {
                                    const headers: HeadersInit = {}
                                    if (token) headers['Authorization'] = `Bearer ${token}`
                                    const res = await fetch(`/sessions/${s.id}`, { method: 'DELETE', headers })
                                    if (!res.ok) {
                                      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
                                      setStatusMessage(err.detail || 'Failed to delete project')
                                    } else {
                                      if (activeSession === s.id) handleUnload()
                                      setStatusMessage(`Project "${s.id}" deleted`)
                                      connectToServer()
                                    }
                                  } catch (err: any) {
                                    setStatusMessage(`Delete failed: ${err?.message || err}`)
                                  }
                                }}
                              ><Trash2 size={14} /></button>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </nav>
                {connected && user && (user.role === 'admin' || user.role === 'manager') && (
                  <div className="bim-actions" style={{ marginTop: 'auto' }}>
                    {!creatingProject ? (
                      <button className="bim-action-btn upload" onClick={() => setCreatingProject(true)}>
                        + New Project
                      </button>
                    ) : (
                      <div className="session-create-input" style={{ display: 'flex', gap: '4px' }}>
                        <input
                          autoFocus
                          placeholder="project-name"
                          value={newProjectName}
                          style={{ flex: 1 }}
                          onChange={e => setNewProjectName(e.target.value)}
                          onKeyDown={async e => {
                            if (e.key === 'Enter' && newProjectName.trim()) {
                              try {
                                const headers: HeadersInit = { 'Content-Type': 'application/json' }
                                if (token) headers['Authorization'] = `Bearer ${token}`
                                const res = await fetch('/sessions', {
                                  method: 'POST', headers,
                                  body: JSON.stringify({ name: newProjectName.trim() }),
                                })
                                if (!res.ok) {
                                  const err = await res.json().catch(() => ({}))
                                  setStatusMessage(err.detail || 'Failed to create project')
                                } else {
                                  setNewProjectName('')
                                  setCreatingProject(false)
                                  connectToServer()
                                }
                              } catch { setStatusMessage('Failed to create project') }
                            }
                            if (e.key === 'Escape') { setCreatingProject(false); setNewProjectName('') }
                          }}
                        />
                        <button className="session-create-confirm"
                          disabled={!newProjectName.trim()}
                          onClick={async () => {
                            if (!newProjectName.trim()) return
                            try {
                              const headers: HeadersInit = { 'Content-Type': 'application/json' }
                              if (token) headers['Authorization'] = `Bearer ${token}`
                              const res = await fetch('/sessions', {
                                method: 'POST', headers,
                                body: JSON.stringify({ name: newProjectName.trim() }),
                              })
                              if (!res.ok) {
                                const err = await res.json().catch(() => ({}))
                                setStatusMessage(err.detail || 'Failed to create project')
                              } else {
                                setNewProjectName('')
                                setCreatingProject(false)
                                connectToServer()
                              }
                            } catch { setStatusMessage('Failed to create project') }
                          }}>✓</button>
                        <button className="session-create-cancel"
                          onClick={() => { setCreatingProject(false); setNewProjectName('') }}>✕</button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}



            {/* Segments Panel */}
            {activePanel === 'segments' && activeSession && (
              <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
                <div className="panel-header">
                  Segments
                  {segments.length > 0 && (
                    <span style={{ float: 'right', display: 'flex', gap: '4px' }}>
                      <button className="segment-toggle-btn" title="Select All"
                        onClick={() => {
                          setSegments(prev => prev.map(s => ({ ...s, visible: true })))
                          setUnsegmentedVisible(true)
                          segments.forEach(s => {
                            viewportRef.current?.toggleOBB(s.key, true)
                            viewportRef.current?.setSegmentVisibility(s.id, true)
                          })
                          viewportRef.current?.setSegmentVisibility(0, true)  // unsegmented
                        }}><CheckSquare size={13} /></button>
                      <button className="segment-toggle-btn" title="Deselect All"
                        onClick={() => {
                          setSegments(prev => prev.map(s => ({ ...s, visible: false })))
                          setUnsegmentedVisible(false)
                          segments.forEach(s => {
                            viewportRef.current?.toggleOBB(s.key, false)
                            viewportRef.current?.setSegmentVisibility(s.id, false)
                          })
                          viewportRef.current?.setSegmentVisibility(0, false)  // unsegmented
                        }}><Square size={13} /></button>
                    </span>
                  )}
                </div>
                <div className="bim-search">
                  <span className="bim-search-icon"><Search size={12} /></span>
                  <input
                    className="bim-search-input"
                    placeholder="Search..."
                    value={segSearch}
                    onChange={e => setSegSearch(e.target.value)}
                  />
                  {segSearch && (
                    <span className="bim-search-clear" onClick={() => setSegSearch('')}>✕</span>
                  )}
                </div>
                {floorLevel.candidates.length > 0 && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 10px', borderBottom: '1px solid var(--border)', fontSize: 12 }}>
                    <span title="Selected floor is leveled to y=0 on the XZ plane" style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>Floor @ y=0</span>
                    <select
                      value={floorLevel.selected ?? ''}
                      onChange={e => {
                        const iid = parseInt(e.target.value)
                        if (!Number.isNaN(iid) && activeSession) {
                          // applies immediately — no confirmation by design
                          applyFloorLevel(activeSession, 'explicit', iid)
                        }
                      }}
                      style={{ flex: 1, background: 'var(--bg-input, #222)', color: 'var(--text-primary)', border: '1px solid var(--border, #444)', borderRadius: 4, padding: '2px 6px', fontSize: 12 }}
                    >
                      {floorLevel.selected == null && <option value="">(auto: lowest)</option>}
                      {floorLevel.candidates.map(c => (
                        <option key={c.instance_id} value={c.instance_id}>
                          {c.label} #{c.instance_id}{c.height_m != null ? ` (y=${c.height_m.toFixed(2)}m)` : ''}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
                <div className="segments-list" style={{ flex: 1, overflowY: 'auto' }}>
                  {segments.length === 0 && (
                    <div style={{ padding: '14px 16px 6px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '12px', opacity: 0.6 }}>
                      No segmented objects yet — the full cloud is listed as “Unsegmented” below.
                    </div>
                  )}
                      {segments.filter(seg => !segSearch || seg.label.toLowerCase().includes(segSearch.toLowerCase())).map(seg => (
                        <div key={seg.key} className="segment-item">
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', width: '100%' }}>
                            <input
                              type="checkbox"
                              checked={seg.visible}
                              title="Toggle Visibility in 3D"
                              className="segment-checkbox"
                              onChange={() => {
                                const newVis = !seg.visible
                                setSegments(prev => prev.map(s =>
                                  s.key === seg.key ? { ...s, visible: newVis } : s
                                ))
                                viewportRef.current?.toggleOBB(seg.key, newVis)
                                viewportRef.current?.setSegmentVisibility(seg.id, newVis)
                              }}
                            />
                            <span
                              className="segment-color-dot"
                              style={{ background: seg.color }}
                            />
                            {editingSegKey === seg.key ? (
                              <input
                                autoFocus
                                defaultValue={seg.label}
                                className="segment-label"
                                style={{ flex: 1, background: 'var(--bg-input)', border: '1px solid var(--accent)', borderRadius: '3px', padding: '1px 4px', color: 'var(--text-primary)', fontSize: '12px', outline: 'none' }}
                                onKeyDown={async (e) => {
                                  if (e.key === 'Enter') {
                                    const newLabel = (e.target as HTMLInputElement).value.trim()
                                    if (newLabel && newLabel !== seg.label && activeSession) {
                                      await fetch('/api/segmentation/rename', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ session_id: activeSession, instance_id: seg.id, label: newLabel, old_label: seg.label.replace(/ #\d+$/, '') }),
                                      })
                                      setSegments(prev => prev.map(s =>
                                        s.key === seg.key ? { ...s, label: newLabel } : s
                                      ))
                                      viewportRef.current?.refreshSegmentOBBs(activeSession)
                                    }
                                    setEditingSegKey(null)
                                  } else if (e.key === 'Escape') {
                                    setEditingSegKey(null)
                                  }
                                }}
                                onBlur={async (e) => {
                                  const newLabel = e.target.value.trim()
                                  if (newLabel && newLabel !== seg.label && activeSession) {
                                    await fetch('/api/segmentation/rename', {
                                      method: 'POST',
                                      headers: { 'Content-Type': 'application/json' },
                                      body: JSON.stringify({ session_id: activeSession, instance_id: seg.id, label: newLabel, old_label: seg.label.replace(/ #\d+$/, '') }),
                                    })
                                    setSegments(prev => prev.map(s =>
                                      s.key === seg.key ? { ...s, label: newLabel } : s
                                    ))
                                    viewportRef.current?.refreshSegmentOBBs(activeSession)
                                  }
                                  setEditingSegKey(null)
                                }}
                              />
                            ) : (
                              <span className="segment-label" style={{ flex: 1 }}>{seg.label}</span>
                            )}
                            <span className="segment-count">
                              ({seg.totalPoints.toLocaleString()})
                            </span>
                            <button className="seg-inst-btn seg-inst-edit" title="Rename"
                              onClick={() => setEditingSegKey(seg.key)}>
                              <Pencil size={12} />
                            </button>
                            <button className="seg-inst-btn seg-inst-del" title="Delete"
                              onClick={async () => {
                                const ok = await confirmDanger(
                                  'Delete Segment',
                                  `Delete "${seg.label}"? This will remove the segment and its masks permanently.`
                                )
                                if (!ok || !activeSession) return
                                const res = await fetch('/api/segmentation/delete', {
                                  method: 'POST',
                                  headers: { 'Content-Type': 'application/json' },
                                  // label pins the EXACT row: instance_id alone
                                  // collides across writers and deleted the
                                  // wrong instance (strip the ' #N' suffix
                                  // some list writers append to the label)
                                  body: JSON.stringify({ session_id: activeSession, instance_id: seg.id, label: seg.label.replace(/ #\d+$/, '') }),
                                })
                                if (res.ok) {
                                  setSegments(prev => prev.filter(s => s.key !== seg.key))
                                  // hide this segment's POINTS immediately — the
                                  // shader keeps painting them until a full cloud
                                  // reload otherwise (delete looked like a no-op)
                                  viewportRef.current?.setSegmentVisibility(seg.id, false)
                                  viewportRef.current?.refreshSegmentOBBs(activeSession)
                                } else {
                                  setStatusMessage(`Delete failed (${res.status}) — segment kept`)
                                }
                              }}>
                              <Trash2 size={12} />
                            </button>
                          </div>
                        </div>
                      ))}
                      {/* Unsegmented points toggle */}
                      <div className="segment-item" style={{ borderTop: '1px solid var(--border)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <input
                            type="checkbox"
                            checked={unsegmentedVisible}
                            title="Show/hide unsegmented points"
                            className="segment-checkbox"
                            onChange={(e) => {
                              setUnsegmentedVisible(e.target.checked)
                              viewportRef.current?.setSegmentVisibility(0, e.target.checked)
                            }}
                          />
                          <span
                            className="segment-color-dot"
                            style={{ background: 'var(--text-muted)' }}
                          />
                          <span className="segment-label" style={{ flex: 1, fontStyle: 'italic', opacity: 0.7 }}>Unsegmented</span>
                        </div>
                      </div>
                  {/* SHAPE section — generated MeshFlow meshes (visual, non-metric) (visibility toggles) */}
                  {shapeMeshes.length > 0 && (
                    <>
                      <div style={{
                        marginTop: '8px',
                        padding: '6px 8px 4px',
                        fontSize: '10px',
                        fontWeight: 600,
                        letterSpacing: '0.08em',
                        color: 'var(--text-secondary)',
                        textTransform: 'uppercase',
                        borderTop: '1px solid var(--border)',
                        opacity: 0.85,
                      }}>
                        🧊 Shape ({shapeMeshes.length})
                      </div>
                      {shapeMeshes.map(m => (
                        <div key={`shape-${m.folder}`} className="segment-item">
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', width: '100%' }}>
                            <input
                              type="checkbox"
                              checked={m.visible}
                              title="Toggle generated mesh visibility"
                              className="segment-checkbox"
                              onChange={() => {
                                const newVis = !m.visible
                                setShapeMeshes(prev => prev.map(x =>
                                  x.folder === m.folder ? { ...x, visible: newVis } : x
                                ))
                                viewportRef.current?.setShapeVisibility(m.instanceId, newVis)
                              }}
                            />
                            <span className="segment-label" style={{ flex: 1 }}>{m.label}</span>
                          </div>
                        </div>
                      ))}
                    </>
                  )}
                  {/* TSDF section — generated TSDF meshes (visibility toggles) */}
                  {tsdfMeshes.length > 0 && (
                    <>
                      <div style={{
                        marginTop: '8px',
                        padding: '6px 8px 4px',
                        fontSize: '10px',
                        fontWeight: 600,
                        letterSpacing: '0.08em',
                        color: 'var(--text-secondary)',
                        textTransform: 'uppercase',
                        borderTop: '1px solid var(--border)',
                        opacity: 0.85,
                      }}>
                        🧱 TSDF ({tsdfMeshes.length})
                      </div>
                      {tsdfMeshes.map(m => (
                        <div key={`tsdf-${m.folder}`} className="segment-item">
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', width: '100%' }}>
                            <input
                              type="checkbox"
                              checked={m.visible}
                              title="Toggle TSDF mesh visibility"
                              className="segment-checkbox"
                              onChange={() => {
                                const newVis = !m.visible
                                setTsdfMeshes(prev => prev.map(x =>
                                  x.folder === m.folder ? { ...x, visible: newVis } : x
                                ))
                                viewportRef.current?.setTsdfVisibility(m.folder, newVis)
                              }}
                            />
                            <span className="segment-label" style={{ flex: 1 }}>{m.label}</span>
                          </div>
                        </div>
                      ))}
                    </>
                  )}
                </div>
                <div className="bim-actions" style={{ marginTop: 'auto', display: 'flex', gap: 4 }}>
                  <button className="bim-action-btn upload" style={{ flex: 1 }}
                    onClick={() => setInteractiveSessionId(activeSession)}>
                    <Crosshair size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Segmentation
                  </button>
                  <button className="bim-action-btn upload" style={{ flex: 1 }}
                    disabled={!activeSession}
                    title="Per-object meshing: Object (generative) or Mesh (RANSAC + Poisson)"
                    onClick={openTsdfModal}>
                    🧩 Meshing
                  </button>
                </div>
              </div>
            )}

            {/* BIM Navigator Panel */}
            {activePanel === 'bim' && (
              <>
                <BIMNavigator
                  models={bimModels}
                  userRole={activeSession ? (user?.role || 'viewer') : 'viewer'}
                  onToggleVisibility={(meshNames, visible) => viewportRef.current?.toggleBIMVisibility(meshNames, visible)}
                  onSelectElement={(meshNames) => viewportRef.current?.highlightBIMElement(meshNames)}
                  onSetOpacity={(meshNames, opacity) => viewportRef.current?.setBIMOpacity(meshNames, opacity)}
                  onUploadIFC={async (file) => {
                    if (!activeSession || !token) return
                    const formData = new FormData()
                    formData.append('file', file)
                    try {
                      const res = await fetch(`/api/sessions/${activeSession}/bim/upload`, {
                        method: 'POST',
                        headers: { 'Authorization': `Bearer ${token}` },
                        body: formData,
                      })
                      if (res.ok) {
                        setStatusMessage(`BIM uploaded: ${file.name}, loading...`)
                        const { loadIFC } = await import('./components/IFCLoader')
                        const url = `/api/sessions/${activeSession}/bim/${file.name}`
                        const result = await loadIFC(url, file.name)
                        viewportRef.current?.addBIMGroup(result.group)
                        setBimModels(prev => [...prev, result])
                        setStatusMessage(`BIM loaded: ${file.name} (${result.group.children.length} elements)`)
                        connectToServer()
                      } else {
                        setStatusMessage(`Upload failed: ${(await res.json()).detail}`)
                      }
                    } catch (err: any) {
                      setStatusMessage(`Upload error: ${err.message}`)
                    }
                  }}
                  onDeleteIFC={async (filename) => {
                    if (!activeSession || !token) return
                    const ok = await confirmDanger(`Delete ${filename}?`, 'Delete BIM File')
                    if (!ok) return
                    try {
                      const res = await fetch(`/api/sessions/${activeSession}/bim/${filename}`, {
                        method: 'DELETE',
                        headers: { 'Authorization': `Bearer ${token}` },
                      })
                      if (res.ok) {
                        setStatusMessage(`BIM deleted: ${filename}`)
                        viewportRef.current?.removeBIMGroup(filename)
                        setBimModels(prev => prev.filter(m => m.filename !== filename))
                        connectToServer()
                      } else {
                        setStatusMessage(`Delete failed: ${(await res.json()).detail}`)
                      }
                    } catch (err: any) {
                      setStatusMessage(`Delete error: ${err.message}`)
                    }
                  }}
                />
                {/* Registration toggle moved to comparison dialog */}
              </>
            )}

            {/* Team Panel */}
            {activePanel === 'team' && (
              <TeamPanel
                onCallUser={(userId, username) => setCallTarget({ userId, username })}
              />
            )}

            {/* Analysis Panel (Sábana) */}
            {activePanel === 'analysis' && sabanaFullMeta && activeSession && (
              <BIMAnalysisPanel meta={sabanaFullMeta} sessionId={activeSession} />
            )}

          </div>
          {/* Resize handle */}
          <div
            className="panel-resize-handle"
            onMouseDown={(e) => {
              e.preventDefault()
              const startX = e.clientX
              const startW = sidebarWidth
              const onMove = (ev: MouseEvent) => {
                const delta = ev.clientX - startX
                const newW = Math.max(180, Math.min(600, startW + delta))
                setSidebarWidth(newW)
              }
              const onUp = () => {
                document.removeEventListener('mousemove', onMove)
                document.removeEventListener('mouseup', onUp)
                document.body.style.cursor = ''
                document.body.style.userSelect = ''
              }
              document.addEventListener('mousemove', onMove)
              document.addEventListener('mouseup', onUp)
              document.body.style.cursor = 'col-resize'
              document.body.style.userSelect = 'none'
            }}
          />
        </div>
      )}

      {/* ── Main Content ── */}
      <main className="main-content">
        {/* Toolbar — visible whenever a session is loaded (any content: cloud,
            IFC, TSDF, etc.). `pointCount > 0` was the old gate, but it tracks
            currently-rendered points (frustum-culled), so it hid the toolbar
            whenever the cloud left the view or was toggled off. */}
        {hasSession && !sessionLoading && (
          <div className="toolbar">
            {!sabanaVisible && (
              <>
              <div className="toolbar-group">
                <button className={`tool-btn ${activeTool === 'navigate' ? 'active' : ''}`}
                  onClick={() => setActiveTool('navigate')} title="Navigate"><RotateCcw size={16} /></button>
                <button className={`tool-btn ${activeTool === 'measure-distance' ? 'active' : ''}`}
                  onClick={() => setActiveTool('measure-distance')} title="Measure Distance"><Ruler size={16} /></button>
                <button className={`tool-btn ${activeTool === 'measure-angle' ? 'active' : ''}`}
                  onClick={() => setActiveTool('measure-angle')} title="Measure Angle"><TriangleRight size={16} /></button>
                <button className={`tool-btn ${activeTool === 'section-box' ? 'active' : ''}`}
                  onClick={() => setActiveTool('section-box')} title="Section Box"><Scissors size={16} /></button>
                <button className={`tool-btn ${activeTool === 'align' ? 'active' : ''}`}
                  onClick={() => setActiveTool(activeTool === 'align' ? 'navigate' : 'align')} title="Align Cloud"><Move size={16} /></button>
                <span style={{ position: 'relative', display: 'inline-block' }}>
                  <button className={`tool-btn ${activeTool === 'erase' ? 'active' : ''}`}
                    onClick={() => setActiveTool(activeTool === 'erase' ? 'navigate' : 'erase')}
                    title="Mark zones (right-click) to erase or reassign"><Brush size={16} /></button>
                  {/* Eraser sub-panel (user 2026-08-29): radius + undo live UNDER
                      the eraser button — they are eraser functions, not toolbar
                      tools. Visible only while the eraser is the active tool. */}
                  {activeTool === 'erase' && (
                    <div style={{
                      position: 'absolute', top: 'calc(100% + 6px)', left: 0,
                      zIndex: 60,
                      display: 'flex', flexDirection: 'column', gap: 8,
                      background: 'rgba(24,26,31,0.97)',
                      border: '1px solid rgba(255,255,255,0.14)',
                      borderRadius: 8, padding: '10px 12px',
                      boxShadow: '0 6px 18px rgba(0,0,0,0.45)',
                      whiteSpace: 'nowrap',
                    }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                        title="Brush size (sphere radius / cube half-side)">
                        <button className={`tool-btn ${eraseShape === 'sphere' ? 'active' : ''}`}
                          style={{ fontSize: 12, padding: '2px 7px' }}
                          title="Sphere brush"
                          onClick={() => setEraseShape('sphere')}>⚪</button>
                        <button className={`tool-btn ${eraseShape === 'cube' ? 'active' : ''}`}
                          style={{ fontSize: 12, padding: '2px 7px' }}
                          title="Cube brush (axis-aligned)"
                          onClick={() => setEraseShape('cube')}>⬜</button>
                        <input type="range" min={3} max={150} step={1}
                          value={Math.round(eraseRadius * 100)}
                          onChange={e => setEraseRadius(Number(e.target.value) / 100)}
                          style={{ width: 96, accentColor: '#ff5555' }} />
                        <span style={{ fontSize: 11, minWidth: 44, opacity: 0.85 }}>
                          {Math.round(eraseRadius * 100)} cm
                        </span>
                      </span>
                      {eraseShape === 'cube' && (
                        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                          title="Cube rotation about the vertical axis">
                          <span style={{ fontSize: 11, opacity: 0.7 }}>↻</span>
                          <input type="range" min={0} max={90} step={1}
                            value={eraseYawDeg}
                            onChange={e => setEraseYawDeg(Number(e.target.value))}
                            style={{ width: 150, accentColor: '#ff9955' }} />
                          <span style={{ fontSize: 11, minWidth: 30, opacity: 0.85 }}>
                            {eraseYawDeg}°
                          </span>
                        </span>
                      )}
                      <button className="tool-btn"
                        disabled={eraseMarks === 0}
                        style={{
                          width: '100%', display: 'flex', alignItems: 'center',
                          gap: 6, justifyContent: 'center',
                          background: eraseMarks > 0 ? '#a83232' : undefined,
                          opacity: eraseMarks > 0 ? 1 : 0.5,
                        }}
                        title="Erase the marked zones (visible segments only; points become unsegmented)"
                        onClick={() => viewportRef.current?.commitErase(null, undefined,
                          segments.filter(s => s.visible).map(s => s.id))}>
                        <Eraser size={14} /> <span style={{ fontSize: 11 }}>Erase ({eraseMarks})</span>
                      </button>
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                        <select value={eraseTarget}
                          onChange={e => setEraseTarget(e.target.value)}
                          style={{ flex: 1, fontSize: 11, background: '#1d2026', color: '#ddd', border: '1px solid rgba(255,255,255,0.18)', borderRadius: 5, padding: '4px 6px' }}>
                          <option value="">Reassign to…</option>
                          {segments.map(s => (
                            <option key={s.id} value={String(s.id)}>{s.label}_{s.id}</option>
                          ))}
                          <option value="new">➕ New segment…</option>
                        </select>
                        <button className="tool-btn"
                          disabled={eraseMarks === 0 || eraseTarget === ''}
                          style={{ display: 'flex', alignItems: 'center', gap: 4, opacity: (eraseMarks > 0 && eraseTarget !== '') ? 1 : 0.5 }}
                          title="Assign the marked zones to the chosen segment (includes unsegmented points)"
                          onClick={() => {
                            const visibles = segments.filter(s => s.visible).map(s => s.id)
                            if (eraseTarget === 'new') {
                              const name = window.prompt('New segment name:')
                              if (!name || !name.trim()) return
                              viewportRef.current?.commitErase(null, name.trim(), visibles, unsegmentedVisible)
                            } else {
                              viewportRef.current?.commitErase(Number(eraseTarget), undefined, visibles, unsegmentedVisible)
                            }
                            setEraseTarget('')
                          }}>
                          <Check size={14} /> <span style={{ fontSize: 11 }}>Assign</span>
                        </button>
                      </div>
                      <div style={{ display: 'flex', gap: 6 }}>
                        <button className="tool-btn"
                          disabled={eraseMarks === 0}
                          style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center', opacity: eraseMarks > 0 ? 1 : 0.5 }}
                          title="Remove all marks without erasing"
                          onClick={() => viewportRef.current?.clearEraseMarks()}>
                          <X size={14} /> <span style={{ fontSize: 11 }}>Clear</span>
                        </button>
                        <button className="tool-btn" style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}
                          title="Undo the last applied commit"
                          onClick={async () => {
                            if (!activeSession) return
                            try {
                              const r = await fetch('/api/segmentation/erase/undo', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ session_id: activeSession }),
                              })
                              const d = await r.json()
                              setStatusMessage(d.ok ? `↩ erase undone (${(d.restored || 0).toLocaleString()} pts restored)` : '↩ nothing to undo')
                            } catch { setStatusMessage('↩ undo failed') }
                          }}>
                          <Undo2 size={14} /> <span style={{ fontSize: 11 }}>Undo</span>
                        </button>
                      </div>
                    </div>
                  )}
                </span>
                <button className="tool-btn" onClick={() => viewportRef.current?.clearMeasurements()}
                  title="Clear Measurements"><Trash2 size={16} /></button>
                <button className="tool-btn" onClick={() => { viewportRef.current?.resetSectionBox(); setActiveTool('navigate') }}
                  title="Reset Section Box"><Unlock size={16} /></button>
                <button className="tool-btn" onClick={() => viewportRef.current?.resetCamera()}
                  title="Reset View (Home)"><Home size={16} /></button>
              </div>
              <div className="toolbar-separator" />
              <div className="toolbar-group" style={{ position: 'relative' }}>
                <button className="tool-btn"
                  title="Display settings — Point Size, Detail (LOD budget), Confidence"
                  onClick={() => setShowDisplayMenu(v => !v)}
                  style={showDisplayMenu ? { background: 'var(--accent)' } : undefined}>
                  <SlidersHorizontal size={16} />
                </button>
                {showDisplayMenu && (
                  <>
                    {/* click-outside backdrop */}
                    <div onClick={() => setShowDisplayMenu(false)}
                      style={{ position: 'fixed', inset: 0, zIndex: 90 }} />
                    <div style={{
                      position: 'absolute', top: '100%', left: 0, marginTop: 6, zIndex: 100,
                      background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 6,
                      padding: 12, minWidth: 240, boxShadow: 'var(--shadow-lg)',
                      display: 'flex', flexDirection: 'column', gap: 12,
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span className="control-label" style={{ minWidth: 74 }}>Point Size</span>
                        <input className="control-slider" type="range" style={{ flex: 1 }}
                          min="0.01" max="5" step="0.01" value={pointSize}
                          onChange={e => setPointSize(parseFloat(e.target.value))} />
                        <span className="control-value" style={{ minWidth: 30 }}>{pointSize.toFixed(2)}</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span className="control-label" style={{ minWidth: 74 }}
                          title="Max points rendered at once (LOD budget). Higher = more of the cloud visible, more GPU/RAM.">Detail</span>
                        <input className="control-slider" type="range" style={{ flex: 1 }}
                          min="2" max="40" step="1" value={pointBudget / 1_000_000}
                          onChange={e => setPointBudget(parseFloat(e.target.value) * 1_000_000)} />
                        <span className="control-value" style={{ minWidth: 30 }}>{(pointBudget / 1_000_000).toFixed(0)}M</span>
                      </div>
                      {hasConfidence && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span className="control-label" style={{ minWidth: 74 }}>Confidence</span>
                          <input className="control-slider" type="range" style={{ flex: 1 }}
                            min={0} max={1} step={0.01} value={confidenceThreshold}
                            onChange={e => setConfidenceThreshold(parseFloat(e.target.value))} />
                          <span className="control-value" style={{ minWidth: 30 }}>{confidenceThreshold.toFixed(2)}</span>
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
              </>
            )}
            {/* ── Camera Poses Toggle (only in NUBE mode) ── */}
            {!sabanaVisible && hasCameraPoses && (
              <>
                <div className="toolbar-separator" />
                <div className="toolbar-group">
                  <button className={`tool-btn ${showCameraPoses ? 'active' : ''}`}
                    onClick={() => setShowCameraPoses(v => !v)}
                    title={showCameraPoses ? 'Hide Camera Poses' : 'Show Camera Poses'}>
                    <Camera size={16} />
                  </button>
                </div>
              </>
            )}
            {/* ── Grid & Axes (visible in both sábana and nube) ── */}
            <div className="toolbar-separator" />
            <div className="toolbar-group">
              <button className={`tool-btn ${showGrid ? 'active' : ''}`}
                onClick={() => setShowGrid(v => !v)}
                title={showGrid ? 'Hide Grid' : 'Show Grid'}>
                <Grid3X3 size={16} />
              </button>
              <button className={`tool-btn ${showAxes ? 'active' : ''}`}
                onClick={() => setShowAxes(v => !v)}
                title={showAxes ? 'Hide Axes' : 'Show Axes'}>
                <Axis3D size={16} />
              </button>
            </div>
            {/* ── Sábana / BIM Comparison ── */}
            {bimModels.length > 0 && segments.length > 0 && (
              <>
                <div className="toolbar-separator" />
                <div className="toolbar-group">
                  {!sabanaVisible && (
                    <button className="tool-btn"
                      onClick={handleGenerateComparison}
                      disabled={sabanaLoading}
                      title="Generate BIM vs Scan comparison">
                      {sabanaLoading ? <Loader2 size={16} className="spin" /> : <Scale size={16} />}
                    </button>
                  )}
                  {!sabanaLoading && sessions.find(s => s.id === activeSession)?.hasSabana && (
                    <button className={`tool-btn ${sabanaVisible ? 'active' : ''}`}
                      onClick={handleToggleSabana}
                      disabled={sabanaLoading}
                      title={sabanaVisible ? 'Hide Sábana' : 'Show Sábana'}>
                      <Thermometer size={16} />
                    </button>
                  )}
                  {sabanaVisible && sabanaMetrics && (
                    <span className="toolbar-chip" title="Overall pass rate">
                      {sabanaMetrics.elements?.filter((e: any) => e.status === 'evaluated').length || 0}/{sabanaMetrics.summary?.total_elements || 0} elements
                    </span>
                  )}
                </div>
              </>
            )}
          </div>
        )}

        {/* Synced video↔3D flythrough overlay (shrinks the viewport to the right half) */}
        {flythroughOpen && (
          <SyncPlayer
            sessionId={flythroughOpen}
            viewportRef={viewportRef}
            onClose={() => setFlythroughOpen(null)}
          />
        )}

        {/* 3D Viewport — don't hide during pipeline runs (pipeline panel shows progress) */}
        <div className={`viewport-container${sessionLoading && !(pipelineRunning && pipelineRunning.status === 'running' && pipelineRunning.session_id === activeSession) ? ' viewport-hidden' : ''}${flythroughOpen ? ' viewport-flythrough' : ''}`}>
          <Viewport
            ref={viewportRef}
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
            showAxes={showAxes}
            showGrid={showGrid}
            pipelineRunning={!!pipelineRunning && pipelineRunning.status === 'running'}
            onPointCount={(n) => { setPointCount(n); if (n > 0) lastCloudLoadAtRef.current = Date.now() }}
            onFps={setFps}
            onStatusMessage={setStatusMessage}
            onSegments={setSegments}
            onPipelineProgress={handlePipelineProgress}
            onVolumeChanged={async (params) => {
              // Gizmo edit finished: persist the volume, then re-evaluate its
              // collision state against the scene and tint it accordingly.
              if (!activeSession) return
              try {
                await fetch('/api/scene/volumes/update', {
                  method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ session_id: activeSession, ...params }),
                })
                const er = await fetch('/api/scene/volumes/evaluate', {
                  method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ session_id: activeSession, volume_id: params.volume_id }),
                })
                const ed = await er.json()
                const occ = 1 - (typeof ed.free_fraction === 'number' ? ed.free_fraction : 1)
                viewportRef.current?.setVolumeStatus(params.volume_id,
                  occ < 0.02 ? 'free' : occ < 0.12 ? 'touching' : 'colliding')
              } catch { /* non-fatal: the volume stays its last color */ }
            }}
            onVolumeDeleted={async (volumeId) => {
              if (!activeSession) return
              try {
                await fetch('/api/scene/volumes/delete', {
                  method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ session_id: activeSession, volume_id: volumeId }),
                })
              } catch { /* ignore */ }
            }}
            onHasConfidence={(has) => {
              setHasConfidence(has)
            }}
            onBimLoaded={(models) => {
              setBimModels(models)
              if (models.length > 0) {
                setShowAxes(false)
                setShowGrid(false)
              }
            }}
            onSabanaLoaded={(nPts) => {
              setSabanaLoading(false)
              setSessionLoading(null)
              setStatusMessage(`Sábana: ${nPts?.toLocaleString()} deviation points`)
            }}
            showCameraPoses={showCameraPoses && !sabanaVisible}
            onHasCameraPoses={setHasCameraPoses}
          />

          {/* Session Loading Overlay — hidden when pipeline is running (show pipeline panel instead) */}
          {sessionLoading && !(pipelineRunning && pipelineRunning.status === 'running' && pipelineRunning.session_id === activeSession) && (
            <div className="session-loading-overlay">
              <div className="slo-particles">
                {Array.from({ length: 20 }, (_, i) => (
                  <div key={i} className="slo-dot" />
                ))}
              </div>
              <div className="slo-logo-wrap">
                <div className="slo-logo-ring" />
                <div className="slo-logo-ring" />
                <div className="slo-logo-ring" />
                <img src="/logo.png" alt="STAC Build" className="slo-logo-img" />
              </div>
              <div className="slo-text">
                <div className="slo-title">Loading Session</div>
                <div className="slo-status">
                  <div className="slo-spinner" />
                  <span>{sessionLoading}</span>
                </div>
              </div>
            </div>
          )}

          {/* Deviation Analysis — will be redesigned as toolbar buttons */}

          {/* Pipeline Loading Overlay — animated logo while building, before cloud exists */}
          {pipelineRunning && pipelineRunning.status === 'running' && pipelineRunning.session_id === activeSession && pointCount === 0 && (
            <div className="session-loading-overlay">
              <div className="slo-particles">
                {Array.from({ length: 20 }, (_, i) => (
                  <div key={i} className="slo-dot" />
                ))}
              </div>
              <div className="slo-logo-wrap">
                <div className="slo-logo-ring" />
                <div className="slo-logo-ring" />
                <div className="slo-logo-ring" />
                <img src="/logo.png" alt="STAC Build" className="slo-logo-img" />
              </div>
              <div className="slo-text">
                <div className="slo-title">Building Point Cloud</div>
                <div className="slo-status">
                  <div className="slo-spinner" />
                  <span>{(() => {
                    const stage = pipelineRunning.stages[pipelineRunning.current_stage_idx]
                    return stage ? `${stage.label}: ${stage.message} (${Math.round(stage.pct)}%)` : 'Initializing...'
                  })()}</span>
                </div>
              </div>
            </div>
          )}

          {/* Welcome screen — shown when no session */}
          {!hasSession && !sessionLoading && (
            <div className="welcome-screen">
              <img src="/logo.png" alt="STAC Build" className="welcome-logo-img" />
              <div className="welcome-subtitle">
                {!connected
                  ? 'Connect to a STAC server to browse and visualize your 3D scan sessions.'
                  : 'Select a project from the sidebar.'}
              </div>
              {!connected && (
                <button className="welcome-btn" onClick={connectToServer}>
                  <Plug size={14} /> Connect to Server
                </button>
              )}
              <div className="welcome-hint">
                {connected
                  ? `${sessions.length} project${sessions.length !== 1 ? 's' : ''} available`
                  : 'Default: wss://localhost:8765'}
              </div>
            </div>
          )}

          {/* Viewport info overlay */}
          {hasSession && (
            <div className="viewport-overlay">
              <div className="viewport-info">
                {pointCount > 0 && `${(pointCount / 1000000).toFixed(2)}M points`}
                {fps > 0 && ` · ${fps} fps`}
              </div>
            </div>
          )}

          {/* Console Panel — horizontal bottom overlay */}
          {consoleOpen && (
            <div className="console-panel">
              <div className="console-header">
                <span>Console</span>
                <button className="console-close" onClick={() => setConsoleOpen(false)} title="Close console"><X size={14} /></button>
              </div>
              <div className="console-body">
                {consoleLogs.map((entry, i) => (
                  <div key={i} className={`console-line console-${entry.level}`}>
                    <span className="console-ts">{entry.ts}</span>
                    <span className="console-msg">{entry.msg}</span>
                  </div>
                ))}
                <div ref={consoleEndRef} />
              </div>
            </div>
          )}

          {/* Pipeline progress overlay — only for the active session */}
          {pipelineRunning && pipelineRunning.stages.length > 0 && pipelineRunning.session_id === activeSession && (
            <div className="pipeline-progress-overlay">
              <div className="pipeline-progress-card">
                <div className="pipeline-progress-header">
                  <span>Pipeline {pipelineRunning.status === 'running' ? <Clock size={14} /> : pipelineRunning.status === 'done' ? <CheckCircle2 size={14} color="#3fb950" /> : pipelineRunning.status === 'failed' ? <XCircle size={14} color="#f85149" /> : <Ban size={14} />}</span>
                  {pipelineRunning.status === 'running' && (
                    <button className="pipeline-cancel-btn" onClick={handlePipelineCancel}>Cancel</button>
                  )}
                </div>
                {pipelineRunning.stages.filter(s => s.enabled).map((stage) => (
                  <div key={stage.id} className={`pipeline-stage-row ${stage.status}`}>
                    <span className="pipeline-stage-icon">{stage.icon}</span>
                    <span className="pipeline-stage-label">{stage.label}</span>
                    <div className="pipeline-stage-bar">
                      <div
                        className="pipeline-stage-fill"
                        style={{ width: `${stage.pct}%` }}
                      />
                    </div>
                    <span className="pipeline-stage-pct">{Math.round(stage.pct)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* ── Right-side collapsible AI Assistant dock ─────────────────────
          Always mounted (the conversation survives collapse); the grid column
          animates to 0 when hidden. Works without a session or segmentation —
          the backend falls back to general chat. */}
      <aside className={`assistant-dock ${assistantOpen ? '' : 'closed'}`}>
        {/* Resize handle on the dock's left edge — same drag behaviour as the
            left sidebar's, mirrored (dragging left widens the dock) */}
        {assistantOpen && (
          <div
            className="assistant-resize-handle"
            onMouseDown={(e) => {
              e.preventDefault()
              const startX = e.clientX
              const startW = assistantWidth
              const onMove = (ev: MouseEvent) => {
                const newW = Math.max(260, Math.min(640, startW - (ev.clientX - startX)))
                setAssistantWidth(newW)
              }
              const onUp = () => {
                document.removeEventListener('mousemove', onMove)
                document.removeEventListener('mouseup', onUp)
                document.body.style.cursor = ''
                document.body.style.userSelect = ''
                setAssistantWidth((w) => {
                  localStorage.setItem('stac.assistantWidth', String(w))
                  return w
                })
              }
              document.addEventListener('mousemove', onMove)
              document.addEventListener('mouseup', onUp)
              document.body.style.cursor = 'col-resize'
              document.body.style.userSelect = 'none'
            }}
          />
        )}
        <div className="assistant-dock-header">
          <span className="assistant-dock-title"><Sparkles size={14} /> AI Assistant</span>
          <button className="assistant-dock-close" title="Hide chat" onClick={toggleAssistant}>
            <ChevronRight size={16} />
          </button>
        </div>
        <AssistantPanel sessionId={activeSession} viewport={viewportRef}
          onVlmStatus={setVlmStatus} />
      </aside>

      {/* Pipeline Config Dialog */}
      {
        pipelineDialogOpen && (
          <div className="pipeline-dialog-backdrop" onClick={() => setPipelineDialogOpen(false)}>
            <div className="pipeline-dialog" onClick={e => e.stopPropagation()}>
              <h3>Run Pipeline</h3>
              <p className="pipeline-dialog-session">Session: {pipelineDialogSession}</p>

              {/* Scan Selection */}
              {scansList.length > 0 && (
                <div className="pipeline-dialog-scans">
                  <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>
                    {scansList.length === 1 ? 'Scan to reconstruct:' : `Select scans to reconstruct (${selectedScans.length}/${scansList.length}):`}
                  </p>
                  {scansList.map(scan => {
                    const isPipelineActive = !!(pipelineRunning && (pipelineRunning.status === 'running' || pipelineRunning.status === 'queued') && pipelineRunning.session_id === pipelineDialogSession)
                    const isRebuilding = isPipelineActive && (!pipelineRunning!.scans || pipelineRunning!.scans.includes(scan.key))
                    return (
                      <label key={scan.key} className="pipeline-scan-item" style={isRebuilding ? { opacity: 0.5 } : undefined}>
                        <input
                          type="checkbox"
                          checked={selectedScans.includes(scan.key)}
                          disabled={scansList.length === 1 || isRebuilding}
                          onChange={e => {
                            if (e.target.checked) {
                              setSelectedScans(prev => [...prev, scan.key])
                            } else {
                              setSelectedScans(prev => prev.filter(k => k !== scan.key))
                            }
                          }}
                        />
                        <span className="pipeline-scan-date">{scan.date}</span>
                        {scan.source !== 'default' && <span className="pipeline-scan-source">{scan.source}</span>}
                        <span className="pipeline-scan-frames">{scan.frame_count} frames</span>
                        {scan.has_output && <span className="pipeline-scan-badge">✓ output</span>}
                        {isRebuilding && <span className="pipeline-scan-badge" style={{ color: 'var(--accent)' }}>⚙️ rebuilding</span>}
                      </label>
                    )
                  })}
                </div>
              )}
              <div className="pipeline-dialog-stages">
                <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '10px' }}>
                  Runs the full reconstruction end-to-end: 3D Reconstruction → Cloud Cleaning → TSDF Mesh
                </p>
                {(() => {
                  const partial = scansList.filter(s => selectedScans.includes(s.key) && s.recon_state === 'partial')
                  if (partial.length === 0) return null
                  const totalCached = partial.reduce((n, s) => n + (s.cached_chunks || 0), 0)
                  return (
                    <div style={{ fontSize: '11px', background: 'rgba(46,160,67,0.12)', border: '1px solid rgba(46,160,67,0.4)', color: 'var(--success)', borderRadius: '6px', padding: '8px', margin: '8px 0' }}>
                      ⏸ Reconstrucción incompleta detectada ({totalCached} chunk{totalCached === 1 ? '' : 's'} en cache).
                      Al ejecutar se <b>reanuda</b> y completa lo que falta sin re-procesar lo ya hecho.
                    </div>
                  )
                })()}
                <label className="pipeline-replace-toggle">
                  <input
                    type="checkbox"
                    checked={pipelineReplace}
                    onChange={e => setPipelineReplace(e.target.checked)}
                  />
                  <span>Replace existing outputs</span>
                </label>
                {pipelineReplace && scansList.some(s => selectedScans.includes(s.key) && s.recon_state === 'partial') && (
                  <div style={{ fontSize: '11px', color: 'var(--warning)', marginTop: '4px' }}>
                    ⚠️ Con esto activado se borra el cache y la reconstrucción arranca de cero. Desactivalo para reanudar.
                  </div>
                )}
              </div>
              <div className="pipeline-dialog-actions">
                <button className="pipeline-btn-cancel" onClick={() => setPipelineDialogOpen(false)}>Cancel</button>
                <button className="pipeline-btn-run" onClick={handlePipelineRun} disabled={selectedScans.length === 0}>
                  <Play size={14} /> Run Pipeline
                </button>
              </div>
            </div>
          </div>
        )
      }

      {/* ── Status Bar ── */}
      <div className="statusbar">
        <div className="statusbar-item">
          <span><Circle size={8} fill={serverAlive ? '#3fb950' : '#f85149'} stroke="none" /></span>
          <span>STAC Server</span>
        </div>
        <div className="statusbar-item">
          <span><Circle size={8} fill={connected ? '#3fb950' : '#f85149'} stroke="none" /></span>
          <span>{connected ? 'Connected' : 'Disconnected'}</span>
        </div>
        <div className="statusbar-spacer" />
        <div className="statusbar-item">
          {activeSession ? `Project: ${activeSession}` : 'No project'}
        </div>
        {statusMessage && (
          <div className="statusbar-item statusbar-status">
            {statusMessage}
          </div>
        )}
        <div className="statusbar-item font-mono">
          {pointCount > 0 ? `${pointCount.toLocaleString()} pts` : ''}
        </div>
      </div>
      {/* Admin Panel Overlay */}
      {adminOpen && <AdminPage onClose={() => setAdminOpen(false)} />}

      {/* WebRTC Call Overlay */}
      {
        (callTarget || incomingCall) && user && (
          <WebRTCCall
            wsRef={teamWsRef}
            userId={user.id}
            callTarget={callTarget}
            incomingCall={incomingCall}
            onClose={() => { setCallTarget(null); setIncomingCall(null) }}
            onIncomingHandled={() => setIncomingCall(null)}
          />
        )
      }

      {/* Segmentation Manager Overlay */}
      {
        interactiveSessionId && (
          <SegmentationManager
            sessionId={interactiveSessionId}
            onClose={async (dirty: boolean) => {
              const sid = interactiveSessionId
              setInteractiveSessionId(null)
              if (!sid) return

              if (!dirty) {
                // Nothing changed — just reload cached segmentation for sidebar
                setStatusMessage('Segmentation closed (no changes)')
                try {
                  const segRes = await fetch(`/api/sessions/${sid}/segmentation`)
                  if (segRes.ok) {
                    const data = await segRes.json()
                    if (Array.isArray(data.instances)) {
                      setSegments(data.instances.map((inst: any) => ({
                        key: inst.global_id || `${inst.label}_${inst.instance_id || inst.id}`,
                        id: inst.instance_id || inst.id,
                        label: `${inst.label}`,
                        color: inst.color || '#4fd1ff',
                        totalPoints: inst.total_points || 0,
                        visible: true,
                        excluded: inst.excluded || false,
                        confThreshold: inst.conf_threshold || 0,
                        instanceId: inst.instance_id || inst.id,
                      })))
                    }
                    viewportRef.current?.refreshSegmentOBBs(sid)
                    // Reload Potree so new classification data (classId per point) takes effect
                    viewportRef.current?.sendCommandPreserveCamera({ type: 'load_session', session_id: sid })
                  }
                } catch { /* silent */ }
                return
              }

              // Masks changed — regenerate DBSCAN + OBBs
              setSessionLoading('Refreshing segmentation…')
              setStatusMessage('Refreshing segmentation (DBSCAN)...')
              try {
                const res = await fetch('/api/segmentation/refresh', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ session_id: sid }),
                })
                if (res.ok) {
                  const data = await res.json()
                  setStatusMessage(`Segmentation refreshed: ${data.instances?.length || 0} instances`)
                  if (Array.isArray(data.instances)) {
                    setSegments(data.instances.map((inst: any) => ({
                      key: inst.global_id || `${inst.label}_${inst.instance_id || inst.id}`,
                      id: inst.instance_id || inst.id,
                      label: `${inst.label}`,
                      color: inst.color || '#4fd1ff',
                      totalPoints: inst.total_points || 0,
                      visible: true,
                      excluded: inst.excluded || false,
                    })))
                  }
                  viewportRef.current?.refreshSegmentOBBs(sid)
                  // Floor→y=0 after finalize: level the selected floor (or the
                  // lowest when several) — the manager may have created/edited
                  // floor segments (applies instantly, no confirmation).
                  await applyFloorLevel(sid, 'auto')
                  refreshFloorLevel(sid)
                  // Reload Potree so new classification data (classId per point) takes effect
                  viewportRef.current?.sendCommandPreserveCamera({ type: 'load_session', session_id: sid })
                }
              } catch { /* silent */ }
              finally { setSessionLoading(null) }
            }}
            onUpdate={async () => {
              if (!activeSession) return
              try {
                const res = await fetch(`/api/sessions/${activeSession}/segmentation`)
                if (res.ok) {
                  const data = await res.json()
                  if (Array.isArray(data.instances)) {
                    setSegments(data.instances.map((inst: any) => ({
                      key: inst.global_id || `${inst.label}_${inst.instance_id || inst.id}`,
                      id: inst.instance_id || inst.id,
                      label: `${inst.label}`,
                      color: inst.color || '#4fd1ff',
                      totalPoints: inst.total_points || 0,
                      visible: true,
                      excluded: inst.excluded || false,
                    })))
                  }
                }
                // Also refresh 3D OBBs in the viewport
                viewportRef.current?.refreshSegmentOBBs(activeSession)
              } catch { /* silent */ }
            }}
            onSuccess={(newInstances) => {
              setStatusMessage(`Successfully propagated ${newInstances.length} instances.`)
              setInteractiveSessionId(null)
              // Refresh segments in sidebar
              if (activeSession) {
                viewportRef.current?.sendCommand({ type: 'refresh_segments', session_id: activeSession })
              }
            }}
          />
        )
      }
      {/* BIM Comparison dialog with registration toggle */}
      {compareDialogOpen && (
        <div className="cd-overlay" onClick={(e) => { if (e.target === e.currentTarget) setCompareDialogOpen(false) }}>
          <div className="cd-dialog cd-confirm">
            <div className="cd-header">
              <img src="/logo.png" alt="STAC" className="cd-logo" />
              <span className="cd-app-name">STAC Build</span>
            </div>
            <div className="cd-body">
              <span className="cd-icon">📐</span>
              <div className="cd-content">
                <div className="cd-title">BIM vs Scan Comparison</div>
              </div>
            </div>
            <div className="compare-dialog-option">
              <label className="bim-alignment-toggle">
                <span className="bim-alignment-label">Registration</span>
                <div className="toggle-switch-container">
                  <span className={`toggle-option-label ${!useManualAlignment ? 'active' : ''}`}>Auto</span>
                  <button
                    className={`toggle-switch ${useManualAlignment ? 'on' : ''}`}
                    onClick={() => setUseManualAlignment(!useManualAlignment)}
                  >
                    <span className="toggle-switch-knob" />
                  </button>
                  <span className={`toggle-option-label ${useManualAlignment ? 'active' : ''}`}>Manual</span>
                </div>
              </label>
            </div>
            <div className="cd-actions">
              <button className="cd-btn cd-btn-cancel" onClick={() => setCompareDialogOpen(false)}>Cancel</button>
              <button className="cd-btn cd-btn-confirm" onClick={runComparison}>Run Comparison</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Shape Export Modal ── */}
      {showShapeModal && (
        <div className="admin-overlay" style={{ zIndex: 2000 }}>
          <div className="admin-panel" style={{ maxWidth: 540, maxHeight: '80vh', overflow: 'auto' }}>
            <div className="admin-header">
              <h2>🧊 Shape Export</h2>
              <button className="admin-close" onClick={() => setShowShapeModal(false)}>✕</button>
            </div>
            <div style={{ padding: 16 }}>
              <p style={{ color: 'var(--text-secondary)', marginBottom: 12, fontSize: 13 }}>
                Select which objects to export. Existing PKLs/meshes will be overwritten.
              </p>

              {(() => {
                const phaseBadge = (phase?: string) => {
                  if (!phase || phase === 'pending') return null
                  const styles: Record<string, { bg: string; fg: string; icon: string; label: string }> = {
                    captioning:      { bg: '#9b59b633', fg: '#bd93d3', icon: '📝', label: 'caption' },
                    exporting_pkl:   { bg: '#3498db33', fg: '#5dade2', icon: '📦', label: 'pkl…' },
                    pkl_ready:       { bg: '#e67e2233', fg: '#f0a868', icon: '📦', label: 'pkl' },
                    reconstructing:  { bg: '#f39c1233', fg: '#f4b656', icon: '🔄', label: 'mesh…' },
                    done:            { bg: '#2ecc7133', fg: '#52d68d', icon: '🧊', label: 'done' },
                    error:           { bg: '#e74c3c33', fg: '#ec7063', icon: '❌', label: 'error' },
                  }
                  const s = styles[phase]
                  if (!s) return null
                  return (
                    <span style={{
                      fontSize: 11, background: s.bg, color: s.fg,
                      padding: '1px 6px', borderRadius: 4, whiteSpace: 'nowrap',
                    }}>
                      {s.icon} {s.label}
                    </span>
                  )
                }

                return segments.filter(s => s.label !== 'Unsegmented').map(seg => {
                  const st = shapeStatus[seg.id]
                  const prog = shapeProgress[seg.id]
                  const checked = shapeSelected.has(seg.id)
                  const livePhase = prog?.phase
                  return (
                    <div key={seg.key} style={{
                      display: 'flex', flexDirection: 'column', gap: 6,
                      padding: '10px 12px', marginBottom: 8,
                      background: checked ? 'var(--bg-secondary)' : 'var(--bg-tertiary)',
                      borderRadius: 8,
                      border: `2px solid ${checked ? seg.color + '55' : 'transparent'}`,
                      opacity: checked ? 1 : 0.6,
                      transition: 'all 0.15s',
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <input type="checkbox" checked={checked}
                          disabled={shapeRunning}
                          onChange={() => {
                            setShapeSelected(prev => {
                              const next = new Set(prev)
                              if (next.has(seg.id)) next.delete(seg.id)
                              else next.add(seg.id)
                              return next
                            })
                          }}
                          style={{ cursor: shapeRunning ? 'not-allowed' : 'pointer' }}
                        />
                        <span style={{
                          width: 12, height: 12, borderRadius: '50%',
                          background: seg.color, flexShrink: 0,
                        }} />
                        <strong style={{ flex: 1, color: 'var(--text-primary)', fontSize: 13 }}>{seg.label}</strong>
                        <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                          {seg.totalPoints.toLocaleString()} pts
                        </span>
                        {/* Live badge wins over static status */}
                        {livePhase ? phaseBadge(livePhase) :
                          st?.has_mesh ? phaseBadge('done') :
                          st?.has_pkl ? phaseBadge('pkl_ready') : null}
                      </div>
                      {prog?.elapsed != null && livePhase === 'done' && (
                        <div style={{ fontSize: 11, color: 'var(--text-secondary)', paddingLeft: 24 }}>
                          mesh ready · {prog.elapsed.toFixed(0)}s
                        </div>
                      )}
                      {prog?.error && (
                        <div style={{ fontSize: 11, color: 'var(--error)', paddingLeft: 24 }}>
                          {prog.error.slice(0, 140)}
                        </div>
                      )}
                      {checked && !shapeRunning && (
                        <>
                          {st?.has_mesh && (
                            <div style={{ fontSize: 11, color: 'var(--warning)', padding: '2px 0' }}>
                              ⚠️ This object already has a mesh. Running will overwrite it.
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  )
                })
              })()}

              <label style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '12px 0 4px', color: 'var(--text-secondary)', fontSize: 13 }}>
                <input type="checkbox" checked={shapeAutoReconstruct} disabled={shapeRunning}
                  onChange={e => setShapeAutoReconstruct(e.target.checked)} />
                Generate mesh after export (runs MeshFlow — ~12 s per object on GPU)
              </label>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', margin: '0 0 12px 24px' }}>
                ⚠ Visual asset (generative, non-metric). Architectural classes are routed to the
                metric surface-fit pipeline instead.
              </div>

              {shapeOverall.phase !== 'idle' && (shapeRunning || shapeOverall.phase === 'done' || shapeOverall.phase === 'error') && (
                <div style={{
                  padding: '10px 12px', marginBottom: 12,
                  background: shapeOverall.phase === 'error' ? 'rgba(231,76,60,.1)' :
                              shapeOverall.phase === 'done' ? 'rgba(46,204,113,.1)' :
                              'rgba(52,152,219,.1)',
                  borderRadius: 6, fontSize: 13,
                  border: `1px solid ${shapeOverall.phase === 'error' ? '#e74c3c' :
                          shapeOverall.phase === 'done' ? '#2ecc71' : '#3498db'}`,
                  color: shapeOverall.phase === 'error' ? '#e74c3c' :
                         shapeOverall.phase === 'done' ? '#2ecc71' : '#5dade2',
                }}>
                  {shapeOverall.phase === 'reconstructing' && (
                    <>🔄 Reconstructing mesh {shapeOverall.done || 0}/{shapeOverall.total || 0}…</>
                  )}
                  {shapeOverall.phase === 'exporting_pkl' && (
                    <>📦 Generating PKLs ({shapeOverall.total || 0})…</>
                  )}
                  {shapeOverall.phase === 'done' && (
                    <>✅ All done — {shapeOverall.done || 0}/{shapeOverall.total || 0} mesh(es) reconstructed</>
                  )}
                  {shapeOverall.phase === 'error' && (
                    <>❌ Pipeline error — check the console log for details</>
                  )}
                </div>
              )}

              {shapeResult && !shapeRunning && shapeOverall.phase !== 'reconstructing' && (
                <div style={{
                  padding: '10px 12px', marginBottom: 12,
                  background: 'rgba(46, 204, 113, 0.1)', borderRadius: 6,
                  border: '1px solid #2ecc71', color: '#2ecc71',
                  fontSize: 13,
                }}>
                  Exported {shapeResult.count} PKL{shapeResult.count !== 1 ? 's' : ''}.
                </div>
              )}

              <button
                className="bim-action-btn upload"
                style={{
                  width: '100%', padding: 12, fontWeight: 600, fontSize: 14,
                  opacity: shapeRunning || shapeSelected.size === 0 ? 0.5 : 1,
                }}
                disabled={shapeRunning || shapeSelected.size === 0}
                onClick={async () => {
                  if (!activeSession) return
                  // Synchronous busy guard — `disabled` only kicks in after the
                  // next React render, this gate fires immediately so a stray
                  // second click in the same turn is dropped.
                  if (shapeBusyRef.current) return
                  shapeBusyRef.current = true
                  setShapeRunning(true)
                  setShapeResult(null)
                  setShapeProgress({})
                  setShapeOverall({ phase: 'exporting_pkl', total: shapeSelected.size, done: 0 })
                  try {
                    const res = await fetch('/api/segmentation/shape/export', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        session_id: activeSession,
                        instance_ids: [...shapeSelected],
                        auto_reconstruct: shapeAutoReconstruct,
                      }),
                    })
                    const data = await res.json()
                    if (data.ok) {
                      setShapeResult({ count: data.count, exported: data.exported })
                      setStatusMessage(`Shape: exported ${data.count} segment(s)` +
                        (data.reconstructing ? ' — reconstructing mesh in background' : ''))
                      if (!data.reconstructing) {
                        await refreshShapeStatus()
                        setShapeRunning(false)
                      }
                      // else: keep shapeRunning=true; the polling effect will track it
                    } else {
                      setStatusMessage(`Shape error: ${data.detail || 'Unknown'}`)
                      setShapeRunning(false)
                    }
                  } catch (err: any) {
                    setStatusMessage(`Shape error: ${err.message}`)
                    setShapeRunning(false)
                  } finally {
                    // Always clear the ref so the button can be re-armed once
                    // the run is fully complete (the polling effect handles
                    // setShapeRunning(false) for the reconstruction path).
                    shapeBusyRef.current = false
                  }
                }}
              >
                {shapeRunning
                  ? (shapeOverall.phase === 'reconstructing'
                      ? `🔄 Reconstructing ${shapeOverall.done || 0}/${shapeOverall.total || 0}…`
                      : '⏳ Working…')
                  : `🚀 Start (${shapeSelected.size} object${shapeSelected.size !== 1 ? 's' : ''})`}
              </button>

              {/* Reconstruction v2 — assemble a coherent scene (parametric surfaces,
                  swept solids, boxes, openings, ...) from the segmented cloud. */}
              <button
                className="bim-action-btn"
                style={{ width: '100%', padding: 10, marginTop: 8, fontWeight: 600, fontSize: 13,
                         opacity: reconRunning ? 0.5 : 1 }}
                disabled={reconRunning}
                title="Classify every segment and reconstruct parametric surfaces / swept solids / boxes + openings; free-form objects use the ShapeR meshes. Writes output/scene/scene.json."
                onClick={async () => {
                  if (!activeSession || reconBusyRef.current) return
                  reconBusyRef.current = true
                  setReconRunning(true)
                  setStatusMessage('Reconstruction v2: starting…')
                  try {
                    const res = await fetch(`/api/segmentation/reconstruct/${activeSession}`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({}),
                    })
                    const data = await res.json().catch(() => ({} as any))
                    if (res.status === 409) {
                      setStatusMessage(`Reconstruction: ${data.detail || 'ya hay una reconstrucción corriendo — esperá a que termine'}`)
                      reconBusyRef.current = false
                      setReconRunning(false)
                      return
                    }
                    if (!res.ok || !data.started) {
                      setStatusMessage(`Reconstruction error: ${data.detail || data.error || 'no se pudo iniciar'}`)
                      reconBusyRef.current = false
                      setReconRunning(false)
                      return
                    }
                    // Job is running in the background — the polling effect above
                    // (keyed on reconRunning) drives the status line and reloads
                    // the scene when it finishes, then clears reconRunning.
                  } catch (err: any) {
                    setStatusMessage(`Reconstruction error: ${err?.message ?? err}`)
                    reconBusyRef.current = false
                    setReconRunning(false)
                  }
                }}
              >
                {reconRunning ? '⏳ Assembling scene…' : '🏗️ Reconstruct scene (v2)'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── TSDF Reconstruction Modal ── */}
      {showTsdfModal && (
        <div className="admin-overlay" style={{ zIndex: 2000 }}>
          <div className="admin-panel" style={{ maxWidth: 540, maxHeight: '80vh', overflow: 'auto' }}>
            <div className="admin-header">
              <h2>🧩 Mesh Generation</h2>
              <button className="admin-close" onClick={() => setShowTsdfModal(false)}>✕</button>
            </div>
            <div style={{ padding: 16 }}>
              <p style={{ color: 'var(--text-secondary)', marginBottom: 12, fontSize: 13 }}>
                Select the segments, then choose: <strong>Object</strong> (generative
                reconstruction, visual asset) or <strong>Mesh</strong> (RANSAC fitted
                surface + Poisson from the object's own cloud — both, to compare).
                Existing meshes are overwritten.
              </p>

              {(() => {
                const phaseBadge = (phase?: string) => {
                  if (!phase || phase === 'pending') return null
                  const styles: Record<string, { bg: string; fg: string; icon: string; label: string }> = {
                    starting:     { bg: '#3498db33', fg: '#5dade2', icon: '▶️', label: 'starting' },
                    integrating:  { bg: '#f39c1233', fg: '#f4b656', icon: '🔄', label: 'integrating' },
                    extracting:   { bg: '#9b59b633', fg: '#bd93d3', icon: '🪄', label: 'extracting' },
                    done:         { bg: '#2ecc7133', fg: '#52d68d', icon: '🧱', label: 'done' },
                    error:        { bg: '#e74c3c33', fg: '#ec7063', icon: '❌', label: 'error' },
                  }
                  const s = styles[phase]
                  if (!s) return null
                  return (
                    <span style={{
                      fontSize: 11, background: s.bg, color: s.fg,
                      padding: '1px 6px', borderRadius: 4, whiteSpace: 'nowrap',
                    }}>
                      {s.icon} {s.label}
                    </span>
                  )
                }

                return segments.filter(s => s.label !== 'Unsegmented').map(seg => {
                  const st = tsdfStatus[seg.id]
                  const prog = tsdfProgress[seg.id]
                  const checked = tsdfSelected.has(seg.id)
                  const livePhase = prog?.phase
                  return (
                    <div key={seg.key} style={{
                      display: 'flex', flexDirection: 'column', gap: 6,
                      padding: '10px 12px', marginBottom: 8,
                      background: checked ? 'var(--bg-secondary)' : 'var(--bg-tertiary)',
                      borderRadius: 8,
                      border: `2px solid ${checked ? seg.color + '55' : 'transparent'}`,
                      opacity: checked ? 1 : 0.6,
                      transition: 'all 0.15s',
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <input type="checkbox" checked={checked}
                          disabled={tsdfRunning}
                          onChange={() => {
                            setTsdfSelected(prev => {
                              const next = new Set(prev)
                              if (next.has(seg.id)) next.delete(seg.id)
                              else next.add(seg.id)
                              return next
                            })
                          }}
                          style={{ cursor: tsdfRunning ? 'not-allowed' : 'pointer' }}
                        />
                        <span style={{
                          width: 12, height: 12, borderRadius: '50%',
                          background: seg.color, flexShrink: 0,
                        }} />
                        <strong style={{ flex: 1, color: 'var(--text-primary)', fontSize: 13 }}>{seg.label}</strong>
                        <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                          {seg.totalPoints.toLocaleString()} pts
                        </span>
                        {livePhase ? phaseBadge(livePhase) :
                          st?.has_mesh ? phaseBadge('done') : null}
                      </div>
                      {prog?.elapsed != null && (livePhase === 'done' || livePhase === 'integrating' || livePhase === 'extracting') && (
                        <div style={{ fontSize: 11, color: 'var(--text-secondary)', paddingLeft: 24 }}>
                          {livePhase === 'done' ? `mesh ready · ${prog.elapsed.toFixed(1)}s` : `${prog.elapsed.toFixed(1)}s elapsed`}
                        </div>
                      )}
                      {prog?.error && (
                        <div style={{ fontSize: 11, color: 'var(--error)', paddingLeft: 24 }}>
                          {prog.error.slice(0, 140)}
                        </div>
                      )}
                      {checked && !tsdfRunning && st?.has_mesh && (
                        <div style={{ fontSize: 11, color: 'var(--warning)', padding: '2px 0 0 24px' }}>
                          ⚠️ This object already has a mesh. Running will overwrite it.
                        </div>
                      )}
                    </div>
                  )
                })
              })()}

              {segments.filter(s => s.label !== 'Unsegmented').length === 0 && (
                <div style={{
                  padding: '10px 12px', marginBottom: 4, fontSize: 12.5, lineHeight: 1.5,
                  background: 'var(--bg-tertiary)', borderRadius: 8,
                  color: 'var(--text-secondary)',
                }}>
                  No segmented objects in this session. Run Segmentation first —
                  meshing is per object.
                </div>
              )}

              {tsdfOverall.phase !== 'idle' && (tsdfRunning || tsdfOverall.phase === 'done' || tsdfOverall.phase === 'error') && (
                <div style={{
                  padding: '10px 12px', marginTop: 12, marginBottom: 12,
                  background: tsdfOverall.phase === 'error' ? 'rgba(231,76,60,.1)' :
                              tsdfOverall.phase === 'done' ? 'rgba(46,204,113,.1)' :
                              'rgba(243,156,18,.1)',
                  borderRadius: 6, fontSize: 13,
                  border: `1px solid ${tsdfOverall.phase === 'error' ? '#e74c3c' :
                          tsdfOverall.phase === 'done' ? '#2ecc71' : '#f39c12'}`,
                  color: tsdfOverall.phase === 'error' ? '#e74c3c' :
                         tsdfOverall.phase === 'done' ? '#2ecc71' : '#f4b656',
                }}>
                  {tsdfOverall.phase === 'integrating' && (
                    <>🔄 Integrating depth → mesh {tsdfOverall.done || 0}/{tsdfOverall.total || 0}…</>
                  )}
                  {tsdfOverall.phase === 'done' && (
                    <>✅ All done — {tsdfOverall.done || 0}/{tsdfOverall.total || 0} mesh(es) reconstructed</>
                  )}
                  {tsdfOverall.phase === 'error' && (
                    <>❌ Pipeline error — check the console log for details</>
                  )}
                </div>
              )}

              {/* Two ways to mesh the selected segments (user 2026-08-29):
                  Object = generative MeshFlow asset; Mesh = RANSAC + Poisson
                  (both, cloud-anchored, so they can be compared). */}
              <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
                <button
                  className="bim-action-btn upload"
                  style={{
                    flex: 1, padding: 12, fontWeight: 600, fontSize: 14,
                    opacity: shapeRunning || tsdfRunning || tsdfSelected.size === 0 ? 0.5 : 1,
                  }}
                  disabled={shapeRunning || tsdfRunning || tsdfSelected.size === 0}
                  title="Generative object reconstruction (MeshFlow) — visual asset, not metric"
                  onClick={async () => {
                    if (!activeSession || shapeBusyRef.current) return
                    shapeBusyRef.current = true
                    setShapeRunning(true)
                    setShapeResult(null)
                    setShapeProgress({})
                    setShapeOverall({ phase: 'exporting_pkl', total: tsdfSelected.size, done: 0 })
                    try {
                      const res = await fetch('/api/segmentation/shape/export', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          session_id: activeSession,
                          instance_ids: [...tsdfSelected],
                          auto_reconstruct: true,
                        }),
                      })
                      const data = await res.json()
                      if (data.ok) {
                        setStatusMessage(`Object: reconstructing ${data.count} object(s) in background`)
                        if (!data.reconstructing) setShapeRunning(false)
                      } else {
                        setStatusMessage(`Object error: ${data.detail || 'Unknown'}`)
                        setShapeRunning(false)
                      }
                    } catch (err: any) {
                      setStatusMessage(`Object error: ${err.message}`)
                      setShapeRunning(false)
                    } finally {
                      shapeBusyRef.current = false
                    }
                  }}
                >
                  {shapeRunning
                    ? `🔄 Object ${shapeOverall.done || 0}/${shapeOverall.total || 0}…`
                    : `🧊 Object (${tsdfSelected.size})`}
                </button>
                <button
                  className="bim-action-btn upload"
                  style={{
                    flex: 1, padding: 12, fontWeight: 600, fontSize: 14,
                    opacity: tsdfRunning || shapeRunning || tsdfSelected.size === 0 ? 0.5 : 1,
                  }}
                  disabled={tsdfRunning || shapeRunning || tsdfSelected.size === 0}
                  title="RANSAC fitted surfaces + Poisson from the object's own cloud — both meshes, for comparison"
                  onClick={async () => {
                    if (!activeSession) return
                    setTsdfRunning(true)
                    setTsdfProgress({})
                    setTsdfOverall({ phase: 'surface_fit', total: tsdfSelected.size, done: 0 })
                    try {
                      const res = await fetch('/api/segmentation/tsdf/export', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          session_id: activeSession,
                          instance_ids: [...tsdfSelected],
                        }),
                      })
                      const data = await res.json()
                      if (data.ok) {
                        setStatusMessage(`Mesh: ransac + poisson for ${data.count} object(s) in background`)
                      } else {
                        setStatusMessage(`Mesh error: ${data.detail || 'Unknown'}`)
                        setTsdfRunning(false)
                      }
                    } catch (err: any) {
                      setStatusMessage(`Mesh error: ${err.message}`)
                      setTsdfRunning(false)
                    }
                  }}
                >
                  {tsdfRunning
                    ? `🔄 Mesh ${tsdfOverall.done || 0}/${tsdfOverall.total || 0}…`
                    : `🧩 Mesh (${tsdfSelected.size})`}
                </button>
              </div>

            </div>
          </div>
        </div>
      )}

      {appDialog}
    </div >
  )
}

export default App
