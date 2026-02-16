/**
 * STAC Build — Main Application
 * Hernán Barreto — Ingerop IN3
 */
import { useState, useCallback, useEffect, useRef } from 'react'
import './App.css'
import Viewport, { ViewportHandle, SegmentInstance } from './components/Viewport'

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

type Tool = 'navigate' | 'measure-distance' | 'measure-angle' | 'section-box'

function App() {
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [activeSession, setActiveSession] = useState<string | null>(null)
  const [activeTool, setActiveTool] = useState<Tool>('navigate')
  const [connected, setConnected] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [pointSize, setPointSize] = useState(2.0)
  const [pointCount, setPointCount] = useState(0)
  const [fps, setFps] = useState(0)
  const [openMenu, setOpenMenu] = useState<string | null>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const viewportRef = useRef<ViewportHandle>(null)
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

  const connectToServer = useCallback(async () => {
    try {
      const res = await fetch('/sessions')
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
  }, [])

  const handleSessionClick = useCallback((sessionId: string) => {
    setActiveSession(sessionId)
    setActiveTool('navigate')
  }, [])

  const handleReconstruct = useCallback((sessionId: string) => {
    if (!confirm('¿Reconstruir geometría para esta sesión? Esto puede tardar varios minutos.')) return
    setActiveSession(sessionId)
    setTimeout(() => {
      viewportRef.current?.sendCommand({ type: 'reconstruct_geometry', session_id: sessionId })
      setStatusMessage(`Reconstructing ${sessionId}...`)
    }, 500)
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

  return (
    <div className={`app-layout ${!sidebarOpen ? 'sidebar-collapsed' : ''}`}>
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
                  onClick={() => menuAction(() => { setConnected(false); setSessions([]); setActiveSession(null) })}>
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
                onClick={() => menuAction(() => setSidebarOpen(!sidebarOpen))}>
                {sidebarOpen ? '◀ Hide Sidebar' : '▶ Show Sidebar'}
                <span className="menu-shortcut">Ctrl+B</span>
              </button>
              <div className="menu-separator" />
              <button className="menu-dropdown-item" disabled={!hasSession}>
                🎯 Reset Camera
                <span className="menu-shortcut">Home</span>
              </button>
              <button className="menu-dropdown-item" disabled={!hasSession}>
                🔲 Fullscreen
                <span className="menu-shortcut">F11</span>
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
      </div>

      {/* ── Sidebar ── */}
      {sidebarOpen && (
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="sidebar-header-left">
              <div className="sidebar-logo">S</div>
              <div>
                <div className="sidebar-title">STAC Build</div>
                <div className="sidebar-subtitle">Ingerop IN3</div>
              </div>
            </div>
            <button className="sidebar-collapse-btn"
              onClick={() => setSidebarOpen(false)} title="Collapse sidebar">
              «
            </button>
          </div>

          <nav className="sidebar-nav">
            {/* Connection */}
            {!connected ? (
              <>
                <div className="nav-section">Server</div>
                <div className="nav-item" onClick={connectToServer}>
                  <span className="nav-item-icon">🔌</span>
                  <span className="nav-item-label">Connect to Server</span>
                </div>
              </>
            ) : (
              <>
                {/* Sessions */}
                <div className="nav-section">Sessions</div>
                <div className="session-list">
                  {sessions.map(s => (
                    <div
                      key={s.id}
                      className={`session-item ${activeSession === s.id ? 'active' : ''}`}
                    >
                      <div className={`session-dot ${s.hasCloud ? 'online' : ''}`} />
                      <div className="session-info" onClick={() => handleSessionClick(s.id)}>
                        <div className="session-name">{s.name}</div>
                        <div className="session-meta">
                          {s.frameCount} frames
                          {s.hasCloud && ` · ${s.cloudSizeMb}MB`}
                          {s.hasSegments && ' · 🏷️'}
                        </div>
                      </div>
                      <div className="session-actions">
                        <button className="session-action-btn load"
                          title="Cargar sesión"
                          onClick={(e) => { e.stopPropagation(); handleSessionClick(s.id) }}
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

                {/* Tools (only with active session) */}
                {hasSession && (
                  <>
                    <div className="nav-section">Tools</div>
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
                    <div className={`nav-item ${activeTool === 'section-box' ? 'active' : ''}`}
                      onClick={() => setActiveTool('section-box')}>
                      <span className="nav-item-icon">✂️</span>
                      <span className="nav-item-label">Section Box</span>
                    </div>
                  </>
                )}

                {/* Segments panel */}
                {segments.length > 0 && (
                  <>
                    <div className="nav-section">Segments</div>
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
              </>
            )}
          </nav>

          <div className="sidebar-footer">
            <div className={`connection-dot ${connected ? 'connected' : ''}`} />
            <span className="connection-text">
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </aside>
      )}

      {/* Expand button when sidebar collapsed */}
      {!sidebarOpen && (
        <button className="sidebar-expand-btn"
          onClick={() => setSidebarOpen(true)} title="Show sidebar">
          »
        </button>
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
              <button className={`tool-btn ${activeTool === 'section-box' ? 'active' : ''}`}
                onClick={() => setActiveTool('section-box')} title="Section Box">✂️</button>
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
            onPointCount={setPointCount}
            onFps={setFps}
            onStatusMessage={setStatusMessage}
            onSegments={setSegments}
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
        </div>
      </main>

      {/* ── Status Bar ── */}
      <div className="statusbar">
        <div className="statusbar-item">
          <span>{connected ? '🟢' : '🔴'}</span>
          <span>{connected ? 'STAC Server' : 'Not connected'}</span>
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
    </div>
  )
}

export default App
