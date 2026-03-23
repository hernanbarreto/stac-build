/**
 * STAC Build — Main Application
 * Hernán Barreto — Ingerop IN3
 */
import { useState, useCallback, useEffect, useRef } from 'react'
import type { ReactNode } from 'react'
import './App.css'
import Viewport, { ViewportHandle, SegmentInstance } from './components/Viewport'
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
  Search, Tag, Hammer, Brush, Sparkles, Plug, Upload, Settings, Crosshair,
  Maximize, Monitor, Grid3X3, RotateCcw, Ruler, TriangleRight, Scissors,
  Move, Palette, BookOpen, Keyboard, Info, Users, LogOut, FolderOpen, Axis3D,
  Building2, ArrowUpFromLine, ChevronLeft, ChevronRight, Trash2, Unlock, Play, X,
  Clock, CheckCircle2, XCircle, Ban, Circle, CheckSquare, Square, Check,
  Scale, Thermometer, Loader2, BarChart3, Home, Pencil, Camera,
} from 'lucide-react'

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

type Tool = 'navigate' | 'measure-distance' | 'measure-angle' | 'section-box' | 'align'

function App() {
  const { confirmDanger, dialogElement: appDialog } = useConfirmDialog()
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [activeSession, setActiveSession] = useState<string | null>(null)
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [activeTool, setActiveTool] = useState<Tool>('navigate')
  const [connected, setConnected] = useState(false)
  const [serverAlive, setServerAlive] = useState(false)
  const [activePanel, setActivePanel] = useState<'sessions' | 'segments' | 'bim' | 'team' | 'analysis' | null>('sessions')

  const [bimModels, setBimModels] = useState<IFCLoadResult[]>([])
  const [sidebarWidth, setSidebarWidth] = useState(280)
  const [consoleOpen, setConsoleOpen] = useState(false)
  const [pointSize, setPointSize] = useState(5.0)
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

  const STAGE_DEF: Record<string, { label: string, icon: ReactNode }> = {
    vlm: { label: 'Scene Analysis', icon: <Search size={14} /> },
    sam3: { label: 'Segmentation', icon: <Tag size={14} /> },
    reconstruction: { label: '3D Reconstruction', icon: <Hammer size={14} /> },
    cloudcompy: { label: 'Global Cloud Cleaning', icon: <Brush size={14} /> },
    instance_cleaner: { label: 'Instance Isolation & Erosion', icon: <Sparkles size={14} /> },
  }
  const [pipelineOrder] = useState<string[]>(['reconstruction', 'cloudcompy', 'vlm', 'sam3', 'instance_cleaner'])
  const [pipelineEnabled, setPipelineEnabled] = useState<Record<string, boolean>>({
    reconstruction: true, cloudcompy: true, vlm: true, sam3: true, instance_cleaner: true
  })

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
  const [scansList, setScansList] = useState<{ date: string; source: string; key: string; frame_count: number; has_output: boolean }[]>([])
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
  const [editingSegKey, setEditingSegKey] = useState<string | null>(null)
  const [segSearch, setSegSearch] = useState('')

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
    if (sess && !sess.hasCloud) {
      setStatusMessage('No cloud available. Run Reconstruct first.')
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
                    color: inst.color || '#00d4ff',
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
        setSelectedScans(scans.filter((s: any) => !rebuildingKeys.includes(s.key)).map((s: any) => s.key))
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
        ordered_stages: pipelineOrder,
        stages: pipelineEnabled,
        replace: pipelineReplace,
        scans: selectedScans,
      })
      const label = selectedScans.length === 1 ? selectedScans[0] : `${selectedScans.length} scans`
      setStatusMessage(`Pipeline started for ${pipelineDialogSession} (${label})...`)
    }, 500)
  }, [pipelineDialogSession, pipelineOrder, pipelineEnabled, pipelineReplace, selectedScans])

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
    setPipelineRunning(prev => {
      const merged = { ...newState, scans: prev?.scans }
      return merged
    })
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
    if (newState.status === 'done' || newState.status === 'failed' || newState.status === 'cancelled') {
      setTimeout(() => setPipelineRunning(null), 5000)
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
      style={panelOpen ? { '--sidebar-width': `${sidebarWidth}px` } as React.CSSProperties : undefined}
    >
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
                disabled={sabanaVisible}
                style={sabanaVisible ? { opacity: 0.4, pointerEvents: 'none' } : {}}
                onClick={() => menuAction(() => setActiveTool(activeTool === 'align' ? 'navigate' : 'align'))}>
                <Move size={14} /> Align Cloud
                <span className="menu-shortcut">G</span>
              </button>
              <div className="menu-separator" />
              <div className="menu-dropdown-item" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Palette size={14} /> Point Size
                <input type="range" min="0.5" max="5" step="0.1" value={pointSize}
                  onChange={e => setPointSize(parseFloat(e.target.value))}
                  onClick={e => e.stopPropagation()}
                  style={{ width: 80, accentColor: 'var(--accent)' }} />
                <span style={{ fontSize: 11, color: '#aaa', minWidth: 24 }}>{pointSize.toFixed(1)}</span>
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
            onClick={() => togglePanel('segments')} title="Segments" disabled={segments.length === 0}>
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
        <button className={`activity-btn ${activePanel === 'team' ? 'active' : ''}`}
          onClick={() => togglePanel('team')} title="Team">
          <Users size={18} />
        </button>
        <div className="activity-spacer" />
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
                            <button className="session-action-btn reconstruct"
                              title="Reconstruct Geometry"
                              onClick={(e) => { e.stopPropagation(); handleReconstruct(s.id) }}
                            ><Hammer size={14} /></button>
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
                {user && (user.role === 'admin' || user.role === 'manager') && (
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
            {activePanel === 'segments' && segments.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
                <div className="panel-header">
                  Segments
                  <span style={{ float: 'right', display: 'flex', gap: '4px' }}>
                    <button className="segment-toggle-btn" title="Select All"
                      onClick={() => {
                        setSegments(prev => prev.map(s => ({ ...s, visible: true })))
                        segments.forEach(s => {
                          viewportRef.current?.toggleOBB(s.key, true)
                          viewportRef.current?.setSegmentVisibility(s.id, true)
                        })
                        viewportRef.current?.setSegmentVisibility(0, true)  // unsegmented
                      }}><CheckSquare size={13} /></button>
                    <button className="segment-toggle-btn" title="Deselect All"
                      onClick={() => {
                        setSegments(prev => prev.map(s => ({ ...s, visible: false })))
                        segments.forEach(s => {
                          viewportRef.current?.toggleOBB(s.key, false)
                          viewportRef.current?.setSegmentVisibility(s.id, false)
                        })
                        viewportRef.current?.setSegmentVisibility(0, false)  // unsegmented
                      }}><Square size={13} /></button>
                  </span>
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
                <div className="segments-list" style={{ flex: 1, overflowY: 'auto' }}>
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
                                    body: JSON.stringify({ session_id: activeSession, instance_id: seg.id, label: newLabel }),
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
                                  body: JSON.stringify({ session_id: activeSession, instance_id: seg.id, label: newLabel }),
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
                              body: JSON.stringify({ session_id: activeSession, instance_id: seg.id }),
                            })
                            if (res.ok) {
                              setSegments(prev => prev.filter(s => s.key !== seg.key))
                              viewportRef.current?.refreshSegmentOBBs(activeSession)
                            }
                          }}>
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </div>
                  ))}
                  {/* Unsegmented points toggle */}
                  <div className="segment-item" style={{ borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <input
                        type="checkbox"
                        defaultChecked={true}
                        title="Show/hide unsegmented points"
                        className="segment-checkbox"
                        onChange={(e) => {
                          viewportRef.current?.setSegmentVisibility(0, e.target.checked)
                        }}
                      />
                      <span
                        className="segment-color-dot"
                        style={{ background: '#666' }}
                      />
                      <span className="segment-label" style={{ flex: 1, fontStyle: 'italic', opacity: 0.7 }}>Unsegmented</span>
                    </div>
                  </div>
                </div>
                {activeSession && (
                  <div className="bim-actions" style={{ marginTop: 'auto' }}>
                    <button className="bim-action-btn upload"
                      onClick={() => setInteractiveSessionId(activeSession)}>
                      <Crosshair size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Segmentation
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* BIM Navigator Panel */}
            {activePanel === 'bim' && (
              <>
                <BIMNavigator
                  models={bimModels}
                  userRole={user?.role || 'viewer'}
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
        {/* Toolbar — only with loaded cloud, hide during loading */}
        {hasSession && !sessionLoading && pointCount > 0 && (
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
                <button className="tool-btn" onClick={() => viewportRef.current?.clearMeasurements()}
                  title="Clear Measurements"><Trash2 size={16} /></button>
                <button className="tool-btn" onClick={() => { viewportRef.current?.resetSectionBox(); setActiveTool('navigate') }}
                  title="Reset Section Box"><Unlock size={16} /></button>
                <button className="tool-btn" onClick={() => viewportRef.current?.resetCamera()}
                  title="Reset View (Home)"><Home size={16} /></button>
              </div>
              <div className="toolbar-separator" />
              <div className="toolbar-group">
                <span className="control-label">Point Size</span>
                <input className="control-slider" type="range"
                  min="0.5" max="5" step="0.1" value={pointSize}
                  onChange={e => setPointSize(parseFloat(e.target.value))} />
                <span className="control-value">{pointSize.toFixed(1)}</span>
              </div>
              {hasConfidence && (
                <div className="toolbar-group">
                  <span className="control-label">Confidence</span>
                  <input className="control-slider" type="range"
                    min={0} max={1} step={0.01}
                    value={confidenceThreshold}
                    onChange={e => setConfidenceThreshold(parseFloat(e.target.value))} />
                  <span className="control-value">{confidenceThreshold.toFixed(2)}</span>
                </div>
              )}
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

        {/* 3D Viewport — don't hide during pipeline runs (pipeline panel shows progress) */}
        <div className={`viewport-container${sessionLoading && !(pipelineRunning && pipelineRunning.status === 'running' && pipelineRunning.session_id === activeSession) ? ' viewport-hidden' : ''}`}>
          <Viewport
            ref={viewportRef}
            pointSize={pointSize}
            confidenceThreshold={confidenceThreshold}
            activeSession={activeSession}
            activeTool={activeTool}
            showAxes={showAxes}
            showGrid={showGrid}
            pipelineRunning={!!pipelineRunning && pipelineRunning.status === 'running'}
            onPointCount={setPointCount}
            onFps={setFps}
            onStatusMessage={setStatusMessage}
            onSegments={setSegments}
            onPipelineProgress={handlePipelineProgress}
            onHasConfidence={(has) => {
              setHasConfidence(has)
              if (has) setConfidenceThreshold(0.0)  // Start at 0 = show all
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
                  <span>Pipeline {pipelineRunning.status === 'running' ? <Clock size={14} /> : pipelineRunning.status === 'done' ? <CheckCircle2 size={14} color="#4ade80" /> : pipelineRunning.status === 'failed' ? <XCircle size={14} color="#f87171" /> : <Ban size={14} />}</span>
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

      {/* Pipeline Config Dialog */}
      {
        pipelineDialogOpen && (
          <div className="pipeline-dialog-backdrop" onClick={() => setPipelineDialogOpen(false)}>
            <div className="pipeline-dialog" onClick={e => e.stopPropagation()}>
              <h3>Configure Pipeline</h3>
              <p className="pipeline-dialog-session">Session: {pipelineDialogSession}</p>

              {/* Scan Selection */}
              {scansList.length > 0 && (
                <div className="pipeline-dialog-scans">
                  <p style={{ fontSize: '11px', color: '#888', marginBottom: '6px' }}>
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
                <p style={{ fontSize: '11px', color: '#888', marginBottom: '10px' }}>Pipeline stages (fixed order):</p>
                {pipelineOrder.map((stageId) => {
                  const def = STAGE_DEF[stageId]
                  // CloudCompy is always coupled with Reconstruction — can't be toggled independently
                  const isCoupled = stageId === 'cloudcompy'
                  const isDisabled = isCoupled
                  return (
                    <div key={stageId}
                      style={{ display: 'flex', alignItems: 'center', padding: '8px', background: 'rgba(255,255,255,0.05)', marginBottom: '4px', borderRadius: '6px' }}
                    >
                      <span style={{ marginRight: '10px', color: '#555', fontSize: '12px', width: '16px', textAlign: 'center' }}>•</span>
                      <label className="pipeline-stage-toggle" style={{ margin: 0, padding: 0, flex: 1, background: 'transparent', opacity: isDisabled ? 0.5 : 1 }}>
                        <input
                          type="checkbox"
                          checked={pipelineEnabled[stageId] ?? true}
                          disabled={isDisabled}
                          onChange={e => {
                            const checked = e.target.checked
                            setPipelineEnabled(prev => {
                              const next = { ...prev, [stageId]: checked }
                              // Couple: Reconstruction on → CloudCompy on
                              if (stageId === 'reconstruction') next.cloudcompy = checked
                              return next
                            })
                          }}
                        />
                        <span className="pipeline-stage-check-icon">{def?.icon}</span>
                        <span>{def?.label || stageId}</span>
                        {isCoupled && <span style={{ fontSize: '10px', color: '#666', marginLeft: '8px' }}>(auto with Reconstruction)</span>}
                      </label>
                    </div>
                  )
                })}
                <label className="pipeline-replace-toggle">
                  <input
                    type="checkbox"
                    checked={pipelineReplace}
                    onChange={e => setPipelineReplace(e.target.checked)}
                  />
                  <span>Replace existing outputs</span>
                </label>
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
          <span><Circle size={8} fill={serverAlive ? '#4ade80' : '#f87171'} stroke="none" /></span>
          <span>STAC Server</span>
        </div>
        <div className="statusbar-item">
          <span><Circle size={8} fill={connected ? '#4ade80' : '#f87171'} stroke="none" /></span>
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
                        color: inst.color || '#00d4ff',
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
                      color: inst.color || '#00d4ff',
                      totalPoints: inst.total_points || 0,
                      visible: true,
                      excluded: inst.excluded || false,
                    })))
                  }
                  viewportRef.current?.refreshSegmentOBBs(sid)
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
                      color: inst.color || '#00d4ff',
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
      {appDialog}
    </div >
  )
}

export default App
