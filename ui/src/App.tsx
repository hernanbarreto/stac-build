/**
 * STAC Build — Main Application
 * Hernán Barreto — Ingerop IN3
 */
import { useState, useCallback, useEffect, useRef } from 'react'
import './App.css'
import Viewport, { ViewportHandle, SegmentInstance } from './components/Viewport'
import TeamPanel from './components/TeamPanel'
import WebRTCCall from './components/WebRTCCall'
import { useAuth } from './context/AuthContext'
import LoginPage from './pages/LoginPage'
import AdminPage from './pages/AdminPage'

interface SessionInfo {
  id: string
  name: string
  date: string
  frameCount: number
  chunkCount: number
  hasCloud: boolean
  hasSegments: boolean
  cloudSizeMb: number
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
  status: string
  current_stage_idx: number
  stages: PipelineStageInfo[]
}

type Tool = 'navigate' | 'measure-distance' | 'measure-angle' | 'section-box'

function App() {
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [activeSession, setActiveSession] = useState<string | null>(null)
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [activeTool, setActiveTool] = useState<Tool>('navigate')
  const [connected, setConnected] = useState(false)
  const [serverAlive, setServerAlive] = useState(false)
  const [activePanel, setActivePanel] = useState<'sessions' | 'tools' | 'segments' | 'team' | null>('sessions')
  const [consoleOpen, setConsoleOpen] = useState(false)
  const [pointSize, setPointSize] = useState(2.0)
  const [pointCount, setPointCount] = useState(0)
  const [fps, setFps] = useState(0)
  const [consoleLogs, setConsoleLogs] = useState<{ ts: string; level: string; msg: string }[]>([])
  const consoleEndRef = useRef<HTMLDivElement>(null)
  const [openMenu, setOpenMenu] = useState<string | null>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const viewportRef = useRef<ViewportHandle>(null)
  const [adminOpen, setAdminOpen] = useState(false)

  // Team / WebRTC state
  const teamWsRef = useRef<WebSocket | null>(null)
  const [callTarget, setCallTarget] = useState<{ userId: number; username: string } | null>(null)
  const [incomingCall, setIncomingCall] = useState<{ from: number; username: string; callId: string; media: string } | null>(null)

  // Pipeline state
  const [pipelineDialogOpen, setPipelineDialogOpen] = useState(false)
  const [pipelineDialogSession, setPipelineDialogSession] = useState<string | null>(null)
  const [pipelineStages, setPipelineStages] = useState<Record<string, boolean>>({
    da3: true, cloudcompy: true, vlm: true, sam3: true
  })
  const [pipelineReplace, setPipelineReplace] = useState(true)
  const [pipelineRunning, setPipelineRunning] = useState<PipelineState | null>(null)

  const { user, token, loading: authLoading, logout } = useAuth()

  // Periodic server health check — updates indicator only
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const resp = await fetch('/health', { signal: AbortSignal.timeout(3000) })
        setServerAlive(resp.ok)
      } catch {
        setServerAlive(false)
      }
    }
    checkHealth()
    const id = setInterval(checkHealth, 3000)
    return () => clearInterval(id)
  }, [])

  // When server goes down, clear session state
  const prevAliveRef = useRef(serverAlive)
  useEffect(() => {
    if (prevAliveRef.current && !serverAlive) {
      // Server just went from alive → dead
      setConnected(false)
      setSessions([])
      setActiveSession(null)
      setSelectedSession(null)
      setSegments([])
      setPointCount(0)
      setStatusMessage('Server disconnected')
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
        cloudSizeMb: s.cloud_size_mb || 0,
      }))
      setSessions(sessionList)
      setConnected(true)
      console.log(`[STAC] Connected. ${sessionList.length} sessions found.`)
    } catch (err) {
      console.error('[STAC] Failed to connect:', err)
      setConnected(false)
    }
  }, [token])

  // Select session — highlight only, no cloud loading
  const handleSessionSelect = useCallback((sessionId: string) => {
    setSelectedSession(sessionId)
  }, [])

  // Load session — actually loads the cloud in the viewport
  const handleSessionLoad = useCallback((sessionId: string) => {
    // Clear React state from previous session
    setSegments([])
    setPointCount(0)
    setStatusMessage('')
    setActiveSession(sessionId)
    setSelectedSession(sessionId)
    setActiveTool('navigate')
  }, [])

  const handleReconstruct = useCallback((sessionId: string) => {
    setPipelineDialogSession(sessionId)
    setPipelineDialogOpen(true)
  }, [])

  const handlePipelineRun = useCallback(() => {
    if (!pipelineDialogSession) return
    // Don't load cloud — pipeline completion callback will send the cloud
    setSelectedSession(pipelineDialogSession)
    setPipelineDialogOpen(false)
    setPipelineRunning({ status: 'queued', current_stage_idx: -1, stages: [] })
    setTimeout(() => {
      viewportRef.current?.sendCommand({
        type: 'run_pipeline',
        session_id: pipelineDialogSession,
        stages: pipelineStages,
        replace: pipelineReplace,
      })
      setStatusMessage(`Pipeline started for ${pipelineDialogSession}...`)
    }, 500)
  }, [pipelineDialogSession, pipelineStages, pipelineReplace])

  const handlePipelineCancel = useCallback(() => {
    if (!activeSession) return
    viewportRef.current?.sendCommand({ type: 'cancel_pipeline', session_id: activeSession })
    setPipelineRunning(null)
    setStatusMessage('Pipeline cancelled')
  }, [activeSession])

  const handlePipelineProgress = useCallback((data: Record<string, unknown>) => {
    const state: PipelineState = {
      status: (data.status as string) || 'running',
      current_stage_idx: (data.current_stage_idx as number) ?? -1,
      stages: (data.stages as PipelineStageInfo[]) || [],
    }
    setPipelineRunning(state)
    // Update status bar with current stage info
    const currentStage = state.stages[state.current_stage_idx]
    if (currentStage) {
      setStatusMessage(`${currentStage.icon} ${currentStage.label}: ${currentStage.message} (${Math.round(currentStage.pct)}%)`)
    }
    if (state.status === 'done' || state.status === 'failed' || state.status === 'cancelled') {
      setTimeout(() => setPipelineRunning(null), 5000)
    }
  }, [])

  const handleSegment = useCallback((sessionId: string) => {
    viewportRef.current?.sendCommand({ type: 'set_prompt', prompt: 'auto' })
    setStatusMessage(`Segmenting ${sessionId}...`)
  }, [])

  const handleUnload = useCallback(() => {
    viewportRef.current?.sendCommand({ type: 'cleared' })
    setActiveSession(null)
    setSegments([])
    setPointCount(0)
    setStatusMessage('')
  }, [])

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
    <div className={`app-layout ${!panelOpen ? 'panel-collapsed' : ''}`}>
      {/* ── Menu Bar ── */}
      <div className="menubar" ref={menuRef}>
        <span className="menu-app-title">⚡ STAC Build</span>

        {/* File Menu */}
        <div className="menu-item">
          <button className={`menu-trigger ${openMenu === 'file' ? 'open' : ''}`}
            onClick={() => toggleMenu('file')}>File</button>
          {openMenu === 'file' && (
            <div className="menu-dropdown">
              {!connected ? (
                <button className="menu-dropdown-item"
                  onClick={() => menuAction(connectToServer)}>
                  🔌 Connect to Server
                </button>
              ) : (
                <button className="menu-dropdown-item"
                  onClick={() => menuAction(() => {
                    viewportRef.current?.sendCommand({ type: 'cleared' })
                    setConnected(false)
                    setSessions([])
                    setActiveSession(null)
                    setSelectedSession(null)
                    setSegments([])
                    setPointCount(0)
                    setActiveTool('navigate')
                    setStatusMessage('')
                  })}>
                  ⏏️ Disconnect
                </button>
              )}
              <div className="menu-separator" />
              <button className="menu-dropdown-item" disabled={!hasSession}>
                📤 Export Point Cloud
                <span className="menu-shortcut">Ctrl+E</span>
              </button>
              <div className="menu-separator" />
              <button className="menu-dropdown-item" disabled>
                ⚙️ Settings
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
                {panelOpen ? '◀ Hide Panel' : '▶ Show Panel'}
                <span className="menu-shortcut">Ctrl+B</span>
              </button>
              <div className="menu-separator" />
              <button className="menu-dropdown-item" disabled={!hasSession}
                onClick={() => menuAction(() => viewportRef.current?.resetCamera())}>
                🎯 Reset Camera
                <span className="menu-shortcut">Home</span>
              </button>
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => {
                  if (document.fullscreenElement) document.exitFullscreen()
                  else document.documentElement.requestFullscreen()
                })}>
                🔲 Fullscreen
                <span className="menu-shortcut">F11</span>
              </button>
              <div className="menu-separator" />
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => {
                  setConsoleOpen(prev => !prev)
                })}>
                {consoleOpen ? '🖥️ Console ✓' : '🖥️ Console'}
                <span className="menu-shortcut">Ctrl+`</span>
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
                🔄 Navigate
                <span className="menu-shortcut">V</span>
              </button>
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => setActiveTool('measure-distance'))}>
                📏 Measure Distance
                <span className="menu-shortcut">M</span>
              </button>
              <button className="menu-dropdown-item"
                onClick={() => menuAction(() => setActiveTool('section-box'))}>
                ✂️ Section Box
                <span className="menu-shortcut">X</span>
              </button>
              <div className="menu-separator" />
              <button className="menu-dropdown-item">
                🎨 Point Size
              </button>
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
                📖 Documentation
              </button>
              <button className="menu-dropdown-item" disabled>
                ⌨️ Keyboard Shortcuts
                <span className="menu-shortcut">Ctrl+/</span>
              </button>
              <div className="menu-separator" />
              <button className="menu-dropdown-item" disabled>
                ℹ️ About STAC Build
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
                  👥 User Management
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
                  setPointCount(0)
                  setActiveTool('navigate')
                  setStatusMessage('')
                  logout()
                })}>
                🚪 Logout
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ── Activity Bar ── */}
      <div className="activity-bar">
        <button className={`activity-btn ${activePanel === 'sessions' ? 'active' : ''}`}
          onClick={() => togglePanel('sessions')} title="Sessions">
          📂
          {sessions.length > 0 && <span className="activity-badge">{sessions.length}</span>}
        </button>
        <button className={`activity-btn ${activePanel === 'tools' ? 'active' : ''}`}
          onClick={() => togglePanel('tools')} title="Tools" disabled={!hasSession}>
          🔧
        </button>
        <button className={`activity-btn ${activePanel === 'segments' ? 'active' : ''}`}
          onClick={() => togglePanel('segments')} title="Segments" disabled={segments.length === 0}>
          🏷️
          {segments.length > 0 && <span className="activity-badge">{segments.length}</span>}
        </button>
        <button className={`activity-btn ${activePanel === 'team' ? 'active' : ''}`}
          onClick={() => togglePanel('team')} title="Team">
          👥
        </button>
        <div className="activity-spacer" />
        <button className={`activity-btn ${consoleOpen ? 'active' : ''}`}
          onClick={() => setConsoleOpen(prev => !prev)} title="Console">
          🖥️
        </button>
        <button className="activity-btn" disabled title="Settings">
          ⚙️
        </button>
      </div>

      {/* ── Panel ── */}
      {panelOpen && (
        <div className="panel">
          {/* Sessions Panel */}
          {activePanel === 'sessions' && (
            <>
              <div className="panel-header">Sessions</div>
              <nav className="sidebar-nav">
                {!connected ? (
                  <>
                    <div className="nav-section">Server</div>
                    <div className="nav-item" onClick={connectToServer}>
                      <span className="nav-item-icon">🔌</span>
                      <span className="nav-item-label">Connect to Server</span>
                    </div>
                  </>
                ) : (
                  <div className="session-list">
                    {sessions.map(s => (
                      <div
                        key={s.id}
                        className={`session-item ${selectedSession === s.id ? 'active' : ''} ${activeSession === s.id ? 'loaded' : ''}`}
                      >
                        <div className={`session-dot ${s.hasCloud ? 'online' : ''} ${activeSession === s.id ? 'loaded' : ''}`} />
                        <div className="session-info" onClick={() => handleSessionSelect(s.id)}>
                          <div className="session-name">{s.name}</div>
                          <div className="session-meta">
                            {s.frameCount} frames
                            {s.hasCloud && ` · ${s.cloudSizeMb}MB`}
                            {s.hasSegments && ' · 🏷️'}
                            {activeSession === s.id && ' · ⚡ loaded'}
                          </div>
                        </div>
                        <div className="session-actions">
                          <button className="session-action-btn load"
                            title="Cargar sesión"
                            onClick={(e) => { e.stopPropagation(); handleSessionLoad(s.id) }}
                          >📂</button>
                          <button className="session-action-btn reconstruct"
                            title="Reconstruir geometría"
                            onClick={(e) => { e.stopPropagation(); handleReconstruct(s.id) }}
                          >🔧</button>
                          {activeSession === s.id && (
                            <button className="session-action-btn segment"
                              title="Segmentar objetos"
                              onClick={(e) => { e.stopPropagation(); handleSegment(s.id) }}
                            >🏷️</button>
                          )}
                          {activeSession === s.id && (
                            <button className="session-action-btn unload"
                              title="Descargar sesión"
                              onClick={(e) => { e.stopPropagation(); handleUnload() }}
                            >⏏</button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </nav>
            </>
          )}

          {/* Tools Panel */}
          {activePanel === 'tools' && hasSession && (
            <>
              <div className="panel-header">Tools</div>
              <nav className="sidebar-nav">
                <div className={`nav-item ${activeTool === 'navigate' ? 'active' : ''}`}
                  onClick={() => setActiveTool('navigate')}>
                  <span className="nav-item-icon">🔄</span>
                  <span className="nav-item-label">Navigate</span>
                </div>
                <div className={`nav-item ${activeTool === 'measure-distance' ? 'active' : ''}`}
                  onClick={() => setActiveTool('measure-distance')}>
                  <span className="nav-item-icon">📏</span>
                  <span className="nav-item-label">Measure Distance</span>
                </div>
                <div className={`nav-item ${activeTool === 'measure-angle' ? 'active' : ''}`}
                  onClick={() => setActiveTool('measure-angle')}>
                  <span className="nav-item-icon">📐</span>
                  <span className="nav-item-label">Measure Angle</span>
                </div>
                <div className={`nav-item ${activeTool === 'section-box' ? 'active' : ''}`}
                  onClick={() => setActiveTool('section-box')}>
                  <span className="nav-item-icon">✂️</span>
                  <span className="nav-item-label">Section Box</span>
                </div>
                <div className="nav-section" style={{ marginTop: '16px' }}>Display</div>
                <div className="nav-item" style={{ padding: '4px 12px' }}>
                  <span className="nav-item-icon">🎨</span>
                  <span className="control-label" style={{ marginRight: '8px' }}>Point Size</span>
                  <input className="control-slider" type="range"
                    min="0.5" max="5" step="0.1" value={pointSize}
                    style={{ flex: 1 }}
                    onChange={e => setPointSize(parseFloat(e.target.value))} />
                  <span className="control-value">{pointSize.toFixed(1)}</span>
                </div>
              </nav>
            </>
          )}

          {/* Segments Panel */}
          {activePanel === 'segments' && segments.length > 0 && (
            <>
              <div className="panel-header">
                Segments
                <span style={{ float: 'right', display: 'flex', gap: '4px' }}>
                  <button className="segment-toggle-btn" title="Select All"
                    onClick={() => {
                      setSegments(prev => prev.map(s => ({ ...s, visible: true })))
                      segments.forEach(s => viewportRef.current?.toggleOBB(s.key, true))
                    }}>☑</button>
                  <button className="segment-toggle-btn" title="Deselect All"
                    onClick={() => {
                      setSegments(prev => prev.map(s => ({ ...s, visible: false })))
                      segments.forEach(s => viewportRef.current?.toggleOBB(s.key, false))
                    }}>☐</button>
                </span>
              </div>
              <div className="segments-list">
                {segments.map(seg => (
                  <div key={seg.key} className="segment-item">
                    <input
                      type="checkbox"
                      checked={seg.visible}
                      className="segment-checkbox"
                      onChange={() => {
                        const newVis = !seg.visible
                        setSegments(prev => prev.map(s =>
                          s.key === seg.key ? { ...s, visible: newVis } : s
                        ))
                        viewportRef.current?.toggleOBB(seg.key, newVis)
                      }}
                    />
                    <span
                      className="segment-color-dot"
                      style={{ background: seg.color }}
                    />
                    <span className="segment-label">{seg.label}</span>
                    <span className="segment-count">
                      ({seg.totalPoints.toLocaleString()})
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}

          {/* Team Panel */}
          {activePanel === 'team' && (
            <TeamPanel
              onCallUser={(userId, username) => setCallTarget({ userId, username })}
            />
          )}


        </div>
      )}

      {/* ── Main Content ── */}
      <main className="main-content">
        {/* Toolbar — only with active session */}
        {hasSession && (
          <div className="toolbar">
            <div className="toolbar-group">
              <button className={`tool-btn ${activeTool === 'navigate' ? 'active' : ''}`}
                onClick={() => setActiveTool('navigate')} title="Navigate">🔄</button>
              <button className={`tool-btn ${activeTool === 'measure-distance' ? 'active' : ''}`}
                onClick={() => setActiveTool('measure-distance')} title="Measure Distance">📏</button>
              <button className={`tool-btn ${activeTool === 'measure-angle' ? 'active' : ''}`}
                onClick={() => setActiveTool('measure-angle')} title="Measure Angle">📐</button>
              <button className={`tool-btn ${activeTool === 'section-box' ? 'active' : ''}`}
                onClick={() => setActiveTool('section-box')} title="Section Box">✂️</button>
              <button className="tool-btn" onClick={() => viewportRef.current?.clearMeasurements()}
                title="Clear Measurements">🗑️</button>
              <button className="tool-btn" onClick={() => { viewportRef.current?.resetSectionBox(); setActiveTool('navigate') }}
                title="Reset Section Box">🔓</button>
            </div>
            <div className="toolbar-separator" />
            <div className="toolbar-group">
              <span className="control-label">Point Size</span>
              <input className="control-slider" type="range"
                min="0.5" max="5" step="0.1" value={pointSize}
                onChange={e => setPointSize(parseFloat(e.target.value))} />
              <span className="control-value">{pointSize.toFixed(1)}</span>
            </div>
          </div>
        )}

        {/* 3D Viewport */}
        <div className="viewport-container">
          <Viewport
            ref={viewportRef}
            pointSize={pointSize}
            activeSession={activeSession}
            activeTool={activeTool}
            onPointCount={setPointCount}
            onFps={setFps}
            onStatusMessage={setStatusMessage}
            onSegments={setSegments}
            onPipelineProgress={handlePipelineProgress}
          />

          {/* Welcome screen — shown when no session */}
          {!hasSession && (
            <div className="welcome-screen">
              <div className="welcome-logo">S</div>
              <div className="welcome-title">STAC Build</div>
              <div className="welcome-subtitle">
                {!connected
                  ? 'Connect to a STAC server to browse and visualize your 3D scan sessions.'
                  : 'Select a session from the sidebar to load the point cloud.'}
              </div>
              {!connected && (
                <button className="welcome-btn" onClick={connectToServer}>
                  🔌 Connect to Server
                </button>
              )}
              <div className="welcome-hint">
                {connected
                  ? `${sessions.length} session${sessions.length !== 1 ? 's' : ''} available`
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
                <button className="console-close" onClick={() => setConsoleOpen(false)} title="Close console">✕</button>
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

          {/* Pipeline progress overlay */}
          {pipelineRunning && pipelineRunning.stages.length > 0 && (
            <div className="pipeline-progress-overlay">
              <div className="pipeline-progress-card">
                <div className="pipeline-progress-header">
                  <span>Pipeline {pipelineRunning.status === 'running' ? '⏳' : pipelineRunning.status === 'done' ? '✅' : pipelineRunning.status === 'failed' ? '❌' : '🚫'}</span>
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
      {pipelineDialogOpen && (
        <div className="pipeline-dialog-backdrop" onClick={() => setPipelineDialogOpen(false)}>
          <div className="pipeline-dialog" onClick={e => e.stopPropagation()}>
            <h3>Configure Pipeline</h3>
            <p className="pipeline-dialog-session">Session: {pipelineDialogSession}</p>
            <div className="pipeline-dialog-stages">
              {[
                { id: 'da3', label: '3D Reconstruction', icon: '🔨' },
                { id: 'cloudcompy', label: 'Cloud Cleaning', icon: '🧹' },
                { id: 'vlm', label: 'Scene Analysis', icon: '🔍' },
                { id: 'sam3', label: 'Segmentation', icon: '🏷️' },
              ].map(stage => (
                <label key={stage.id} className="pipeline-stage-toggle">
                  <input
                    type="checkbox"
                    checked={pipelineStages[stage.id] ?? true}
                    onChange={e => setPipelineStages(prev => ({ ...prev, [stage.id]: e.target.checked }))}
                  />
                  <span className="pipeline-stage-check-icon">{stage.icon}</span>
                  <span>{stage.label}</span>
                </label>
              ))}
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
              <button className="pipeline-btn-run" onClick={handlePipelineRun}>▶ Run Pipeline</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Status Bar ── */}
      <div className="statusbar">
        <div className="statusbar-item">
          <span>{serverAlive ? '🟢' : '🔴'}</span>
          <span>STAC Server</span>
        </div>
        <div className="statusbar-item">
          <span>{connected ? '🟢' : '🔴'}</span>
          <span>{connected ? 'Connected' : 'Disconnected'}</span>
        </div>
        <div className="statusbar-spacer" />
        <div className="statusbar-item">
          {activeSession ? `Session: ${activeSession}` : 'No session'}
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
      {(callTarget || incomingCall) && user && (
        <WebRTCCall
          wsRef={teamWsRef}
          userId={user.id}
          callTarget={callTarget}
          incomingCall={incomingCall}
          onClose={() => { setCallTarget(null); setIncomingCall(null) }}
          onIncomingHandled={() => setIncomingCall(null)}
        />
      )}
    </div>
  )
}

export default App
