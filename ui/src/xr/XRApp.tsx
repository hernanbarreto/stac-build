// STAC-Builder — mobile XR viewer (React, design-system styled, responsive).
//
// Session list → XR view on the self-hosted 8th Wall engine: live camera +
// SLAM tracking + reticle on real surfaces + tap-to-place the metric mesh
// (1:10 tabletop first, scale button for 1:1) + measurement tools.
//
// Hernán Barreto - Ingerop IN3 Session IV - STAC

import { useCallback, useEffect, useRef, useState } from 'react'
import { StacXREngine, tele, type Tool } from './engine'

interface ArSession {
  id: string
  has_mesh: boolean
  mesh_bytes: number
  has_cloud: boolean
  cloud_source: string | null
  has_ai: boolean
}

const TOOL_HINTS: Record<Tool, string> = {
  move: 'Aim the circle at the floor and tap to place',
  dist: 'Tap two points on the mesh',
  angle: 'Tap 3 points (vertex second)',
  vol: 'Tap 2 base corners, then a height point',
}

export default function XRApp() {
  const [active, setActive] = useState<ArSession | null>(null)
  return active
    ? <XRView session={active} onBack={() => setActive(null)} />
    : <SessionList onOpen={setActive} />
}

// ── session picker ────────────────────────────────────────────────────────────

function SessionList({ onOpen }: { onOpen: (s: ArSession) => void }) {
  const [sessions, setSessions] = useState<ArSession[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    tele('list-boot', { ua: navigator.userAgent })
    fetch('/api/ar/sessions')
      .then((r) => r.json())
      .then((d) => setSessions(d.sessions ?? []))
      .catch((e) => setError(String(e)))
  }, [])

  return (
    <div className="xr-home">
      <h1>STAC <span className="brand">XR</span></h1>
      <div className="sub">Project a metric reconstruction into your surroundings</div>
      {error && <div className="empty">Backend unreachable: {error}</div>}
      {!error && sessions === null && <div className="empty">Loading sessions…</div>}
      {sessions?.length === 0 && <div className="empty">No reconstructions with a mesh yet.</div>}
      {sessions?.filter((s) => s.has_mesh).map((s) => (
        <button key={s.id} className="session-card" onClick={() => onOpen(s)}>
          <div className="name">{s.id}</div>
          <div className="meta">
            <span className="chip on">mesh {(s.mesh_bytes / 1e6).toFixed(0)} MB</span>
            {s.has_cloud && <span className="chip">{s.cloud_source}</span>}
            {s.has_ai && <span className="chip ai">AI ✦</span>}
          </div>
          <div className="go">Open in XR →</div>
        </button>
      ))}
    </div>
  )
}

// ── XR view ───────────────────────────────────────────────────────────────────

function XRView({ session, onBack }: { session: ArSession, onBack: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<StacXREngine | null>(null)
  const [phase, setPhase] = useState<'arm' | 'starting' | 'ready'>('arm')
  const [tracking, setTracking] = useState('UNSPECIFIED')
  const [error, setError] = useState<string | null>(null)
  const [tool, setTool] = useState<Tool>('move')
  const [scaleLabel, setScaleLabel] = useState('1:10')
  const [toast, setToast] = useState<string | null>(null)
  const toastTimer = useRef<ReturnType<typeof setTimeout>>()

  const say = useCallback((msg: string, ms = 2600) => {
    setToast(msg)
    clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(null), ms)
  }, [])

  useEffect(() => {
    const engine = new StacXREngine(session.id, {
      onReady: () => { setPhase('ready'); say(TOOL_HINTS.move, 4000) },
      onError: (msg) => setError(msg),
      onToast: say,
      onPlaced: () => say('Placed — scale button for 1:1, tools below to measure'),
      onTracking: (status) => setTracking(status),
    })
    engineRef.current = engine
    return () => { engine.stop() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.id])

  // iOS grants camera/motion permission ONLY inside a direct user gesture —
  // auto-starting the engine on mount left the camera dead (black screen).
  const armAR = useCallback(async () => {
    tele('start-tap', {})
    setPhase('starting')
    try {
      const DME: any = (window as any).DeviceMotionEvent
      if (typeof DME?.requestPermission === 'function') {
        const m = await DME.requestPermission().catch((e: any) => `err:${e}`)
        tele('motion-permission', { result: String(m) })
        if (m === 'denied') {
          say('Motion DENIED — without it tracking cannot hold. Close this tab '
            + 'completely, reopen, and tap Allow.', 8000)
        }
      }
      const DOE: any = (window as any).DeviceOrientationEvent
      if (typeof DOE?.requestPermission === 'function') {
        const o = await DOE.requestPermission().catch((e: any) => `err:${e}`)
        tele('orientation-permission', { result: String(o) })
      }
    } catch { /* permission APIs are iOS-only */ }
    engineRef.current?.start(canvasRef.current!)
  }, [])

  const onTap = useCallback((e: React.TouchEvent) => {
    const t = e.changedTouches[0]
    const res = engineRef.current?.tap(t.clientX, t.clientY)
    if (res === 'tracking') say('Tracking not ready — walk a slow step pointing at the floor')
    else if (res === 'no-surface') say('No surface detected — sweep the phone slowly')
    else if (res === 'place-first') say('Place the model first (Move + tap)')
    else if (res === 'no-mesh') say('No mesh under the tap')
  }, [say])

  const pickTool = (t: Tool) => {
    setTool(t)
    engineRef.current?.setTool(t)
    say(TOOL_HINTS[t])
  }

  return (
    <div className="xr-view">
      <canvas ref={canvasRef} className="xr-canvas" onTouchEnd={onTap} />
      <header className="xr-topbar">
        <button className="btn" onClick={() => { engineRef.current?.stop(); onBack() }}>←</button>
        <div className="title">{session.id}</div>
        <div className="spacer" />
        <button className="btn" onClick={() =>
          setScaleLabel(engineRef.current?.setScaleIdx(
            (engineRef.current.scaleIdx + 1)) ?? '1:10')}>
          {scaleLabel}
        </button>
        <button className="btn" title="recenter tracking" onClick={() => {
          engineRef.current?.recenter(); say('Tracking recentered')
        }}>⟲</button>
      </header>
      <nav className="xr-toolbar">
        {(['move', 'dist', 'angle', 'vol'] as Tool[]).map((t) => (
          <button key={t}
                  className={`btn ${tool === t ? 'active' : ''}`}
                  onClick={() => pickTool(t)}>
            {{ move: 'Move', dist: 'Distance', angle: 'Angle', vol: 'Volume' }[t]}
          </button>
        ))}
        <button className="btn" onClick={() => engineRef.current?.clearMeasures()}>
          Clear
        </button>
      </nav>
      {phase === 'ready' && tracking !== 'NORMAL' && (
        <div className="xr-coach">
          📐 Initializing tracking — <strong>walk a slow step</strong> pointing
          the camera at the floor
        </div>
      )}
      {toast && <div className="xr-toast">{toast}</div>}
      {(phase !== 'ready' || error) && (
        <div className="xr-splash">
          {!error && phase === 'starting' && <div className="spin" />}
          <div className="msg">
            {error ?? (phase === 'arm'
              ? <><strong>{session.id}</strong><br />
                  Point the camera at your surroundings and place the metric mesh.</>
              : <>Starting camera + tracking, loading mesh…<br />
                  <small>Allow camera and motion access when asked</small></>)}
          </div>
          {!error && phase === 'arm' && (
            <button className="btn primary-cta" onClick={armAR}>Start AR</button>
          )}
          {error && <button className="btn" onClick={() => location.reload()}>Retry</button>}
        </div>
      )}
    </div>
  )
}
