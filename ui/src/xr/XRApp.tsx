// STAC-Builder — mobile XR viewer (React, design-system styled, responsive).
//
// Session list -> XR view on the self-hosted 8th Wall engine: live camera +
// SLAM tracking + reticle on real surfaces + tap-to-place the metric mesh
// (1:10 tabletop first, scale button for 1:1) + measurement tools.
//
// Hernán Barreto - Ingerop IN3 Session IV - STAC

import { useCallback, useEffect, useRef, useState } from 'react'
import { StacXREngine, tele, type Tool } from './engine'
import { WebXREngine } from './webxr-engine'
import type { IXREngine } from './engine-types'
import { getT, getFmt } from '../i18n'
import { ArrowLeft, RotateCcw, Loader2 } from 'lucide-react'

declare global {
  interface Window { VLaunch?: { getLaunchUrl: (url: string) => Promise<string> } }
}

type XRMode = 'webxr' | 'vlaunch' | 'engine'

interface ArSession {
  id: string
  has_mesh: boolean
  mesh_bytes: number
  has_cloud: boolean
  cloud_source: string | null
  has_ai: boolean
}

const TOOL_HINTS: Record<Tool, string> = { move: 'xr.hintMove', dist: 'xr.hintDist', angle: 'xr.hintAngle', vol: 'xr.hintVol' }
const TOOL_LABELS: Record<Tool, string> = { move: 'xr.toolMove', dist: 'xr.toolDist', angle: 'xr.toolAngle', vol: 'xr.toolVol' }

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
      <h1 className="xr-home__title">{getT()('xr.title')}</h1>
      <div className="sub">{getT()('xr.subtitle')}</div>
      {error && <div className="empty">{getT()('xr.backendUnreachable', { detail: error })}</div>}
      {!error && sessions === null && <div className="empty">{getT()('xr.loadingSessions')}</div>}
      {sessions?.length === 0 && <div className="empty">{getT()('xr.noMeshes')}</div>}
      {sessions?.filter((s) => s.has_mesh).map((s) => (
        <button key={s.id} className="session-card" onClick={() => onOpen(s)}>
          <div className="name">{s.id}</div>
          <div className="meta">
            <span className="chip on">{getT()('xr.meshSize', { mb: getFmt().bytesMb(s.mesh_bytes / 1e6) })}</span>
            {s.has_cloud && <span className="chip">{s.cloud_source}</span>}
            {s.has_ai && <span className="chip ai">{getT()('xr.ai')}</span>}
          </div>
          <div className="go">{getT()('xr.open')}</div>
        </button>
      ))}
    </div>
  )
}

// ── XR view ───────────────────────────────────────────────────────────────────

function XRView({ session, onBack }: { session: ArSession, onBack: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<IXREngine | null>(null)
  const [phase, setPhase] = useState<'arm' | 'starting' | 'ready'>('arm')
  const [mode, setMode] = useState<XRMode | null>(null)
  const [tracking, setTracking] = useState('UNSPECIFIED')
  const [error, setError] = useState<string | null>(null)
  const [tool, setTool] = useState<Tool>('move')
  const [scaleLabel, setScaleLabel] = useState('1:1')   // engine default = metric
  const [toast, setToast] = useState<string | null>(null)
  const toastTimer = useRef<ReturnType<typeof setTimeout>>()

  const say = useCallback((msg: string, ms = 2600) => {
    setToast(msg)
    clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(null), ms)
  }, [])

  // capability probe: real WebXR (Android Chrome natively, or iPhone INSIDE
  // the Variant App Clip = ARKit) -> best; iOS Safari outside the clip ->
  // vlaunch button; anything else -> the 8th Wall engine fallback
  useEffect(() => {
    let alive = true
    ;(async () => {
      const ok = await (navigator as any).xr
        ?.isSessionSupported?.('immersive-ar').catch(() => false)
      const ios = /iPhone|iPad|iPod/.test(navigator.userAgent)
      const m: XRMode = ok ? 'webxr'
        : (ios && window.VLaunch ? 'vlaunch' : 'engine')
      if (alive) { setMode(m); tele('xr-mode', { mode: m, session: session.id }) }
    })()
    return () => { alive = false; engineRef.current?.stop() }
  }, [session.id])

  const mkCallbacks = useCallback(() => ({
    onReady: () => { setPhase('ready'); say(TOOL_HINTS.move, 4000) },
    onError: (msg: string) => setError(msg),
    onToast: say,
    onPlaced: () => say(getT()('xr.placed')),
    onTracking: (status: string) => setTracking(status),
  }), [say])

  // iOS grants camera/motion permission ONLY inside a direct user gesture —
  // auto-starting the engine on mount left the camera dead (black screen).
  const armAR = useCallback(async () => {
    tele('start-tap', { mode })
    if (mode === 'vlaunch') {
      // hand off to the Variant App Clip (real ARKit): the page must live on
      // a PUBLIC domain authorized in the Variant dashboard — App Clips
      // cannot reach private/tailnet URLs (the root cause of the earlier
      // infinite-loading hang)
      try {
        // PREFER the direct Apple App Clip URL (skips Variant's interstitial,
        // which hung repeatedly); fall back to their launch page with the
        // http:// scheme bug fixed (their API omits the s; port 80 refuses).
        const direct = (window.VLaunch as any).directAppClipUrl as string | undefined
        const url = direct
          ?? (await window.VLaunch!.getLaunchUrl(location.href))
            .replace(/^http:\/\//, 'https://')
        tele('vlaunch-navigate', { direct: !!direct, url: url.slice(0, 90) })
        location.href = url
      } catch (e: any) {
        tele('vlaunch-error', { msg: String(e?.message ?? e) })
        say(getT()('xr.arkitFailed', { detail: e?.message ?? e }), 6000)
      }
      return
    }
    setPhase('starting')
    try {
      const DME: any = (window as any).DeviceMotionEvent
      if (typeof DME?.requestPermission === 'function') {
        const m = await DME.requestPermission().catch((e: any) => `err:${e}`)
        tele('motion-permission', { result: String(m) })
      }
      const DOE: any = (window as any).DeviceOrientationEvent
      if (typeof DOE?.requestPermission === 'function') {
        const o = await DOE.requestPermission().catch((e: any) => `err:${e}`)
        tele('orientation-permission', { result: String(o) })
      }
    } catch { /* permission APIs are iOS-only */ }
    const engine: IXREngine = mode === 'webxr'
      ? new WebXREngine(session.id, mkCallbacks())
      : new StacXREngine(session.id, mkCallbacks())
    engineRef.current = engine
    engine.start(canvasRef.current!)
  }, [mode, session.id, mkCallbacks, say])

  const onTap = useCallback((e: React.TouchEvent) => {
    const t = e.changedTouches[0]
    const res = engineRef.current?.tap(t.clientX, t.clientY)
    if (res === 'tracking') say(getT()('xr.trackingNotReady'))
    else if (res === 'no-surface') say(getT()('xr.noSurface'))
    else if (res === 'place-first') say(getT()('xr.placeFirst'))
    else if (res === 'no-mesh') say(getT()('xr.noMesh'))
  }, [say])

  const pickTool = (t: Tool) => {
    setTool(t)
    engineRef.current?.setTool(t)
    say(getT()(TOOL_HINTS[t]))
  }

  return (
    <div className="xr-view">
      <canvas ref={canvasRef} className="xr-canvas" onTouchEnd={onTap} />
      <header className="xr-topbar">
        <button className="btn" aria-label={getT()('xr.back')} onClick={() => { engineRef.current?.stop(); onBack() }}><ArrowLeft aria-hidden /></button>
        <div className="title">{session.id}</div>
        <div className="spacer" />
        <button className="btn" onClick={() =>
          setScaleLabel(engineRef.current?.setScaleIdx(
            (engineRef.current.scaleIdx + 1)) ?? '1:10')}>
          {scaleLabel}
        </button>
        <button className="btn" aria-label={getT()('xr.recenter')} onClick={() => {
          engineRef.current?.recenter(); say(getT()('xr.recentered'))
        }}><RotateCcw aria-hidden /></button>
      </header>
      <nav className="xr-toolbar">
        {(['move', 'dist', 'angle', 'vol'] as Tool[]).map((t) => (
          <button key={t}
                  className={`btn ${tool === t ? 'active' : ''}`}
                  onClick={() => pickTool(t)}>
            {getT()(TOOL_LABELS[t])}
          </button>
        ))}
        <button className="btn" onClick={() => engineRef.current?.clearMeasures()}>
          {getT()('common.clear')}
        </button>
        <button className="btn" onClick={() => {
          const eng = engineRef.current as any
          if (eng?.toggleCloud) say(eng.toggleCloud() ? getT()('xr.cloudOn') : getT()('xr.cloudOff'), 1200)
        }}>
          {getT()('xr.cloud')}
        </button>
      </nav>
      {phase === 'ready' && tracking !== 'NORMAL' && (
        <div className="xr-coach">
          {getT()('xr.initializingTracking')}
        </div>
      )}
      {toast && <div className="xr-toast">{toast}</div>}
      {(phase !== 'ready' || error) && (
        <div className="xr-splash">
          {!error && phase === 'starting' && <Loader2 className="stac-spin xr-spin" aria-hidden />}
          <div className="msg">
            {error ?? (phase === 'arm'
              ? <><strong className="xr-splash__strong">{session.id}</strong><br />{getT()('xr.armHint')}</>
              : <>{getT()('xr.starting')}<br /><small>{getT()('xr.permissionsHint')}</small></>)}
          </div>
          {!error && phase === 'arm' && mode !== null && (
            <button className="btn primary-cta" onClick={armAR}>
              {mode === 'vlaunch' ? getT()('xr.startArkit') : getT()('xr.startAr')}
            </button>
          )}
          {!error && phase === 'arm' && mode === null && <Loader2 className="stac-spin xr-spin" aria-hidden />}
          {error && <button className="btn" onClick={() => location.reload()}>{getT()('common.retry')}</button>}
        </div>
      )}
    </div>
  )
}
