import { useEffect, useRef, useState, useCallback } from 'react'
import { Play, Pause, X, Loader2, SkipBack, SkipForward } from 'lucide-react'
import type { ViewportHandle } from './Viewport'
import { IconButton } from './ui/IconButton'
import { SegmentedControl } from './ui/Field'
import { useFmt, useT } from '../i18n'

interface Flythrough {
  n_frames: number
  frames: number[]           // REAL video frame index of each keyframe pose (ascending, non-uniform)
  poses: number[][]          // per-keyframe c2w (16 floats, row-major), scaled to cloud frame
  intrinsics: number[] | null // [fx, fy, cx, cy] at reconstruction resolution
  fps: number | null          // source video fps (for time->frame mapping)
  video_n_frames: number | null // total frames in the source video
  video_url: string
}

/** The camera path, precomputed once: per keyframe the position and the camera's y and z axes
 *  (OpenCV c2w columns) with their Hermite tangents per video frame — a C1 curve in TIME
 *  (non-uniform Catmull-Rom), so the camera's speed and turn rate never jump at a keyframe. */
interface Path {
  f: Float64Array            // keyframe → video frame
  P: Float64Array; Y: Float64Array; Z: Float64Array        // 3 per keyframe
  mP: Float64Array; mY: Float64Array; mZ: Float64Array     // tangents (per frame)
  n: number
}

function buildPath(d: Flythrough): Path {
  const n = d.poses.length
  const f = new Float64Array(n), P = new Float64Array(3 * n), Y = new Float64Array(3 * n), Z = new Float64Array(3 * n)
  for (let k = 0; k < n; k++) {
    const m = d.poses[k]                         // row-major: m[r*4+c]
    f[k] = d.frames[k] ?? k
    for (let r = 0; r < 3; r++) {
      P[3 * k + r] = m[r * 4 + 3]
      Y[3 * k + r] = m[r * 4 + 1]
      Z[3 * k + r] = m[r * 4 + 2]
    }
  }
  // shape-preserving tangents per component (Fritsch–Butland on non-uniform knots): zero where
  // the motion stops or reverses, so the camera never overshoots a keyframe or drifts while the
  // video stands still (parallax keyframes are dense in motion, sparse when the walker stops)
  const tan = (A: Float64Array) => {
    const out = new Float64Array(3 * n)
    for (let k = 0; k < n; k++) {
      for (let r = 0; r < 3; r++) {
        const h0 = k > 0 ? f[k] - f[k - 1] : 0, h1 = k < n - 1 ? f[k + 1] - f[k] : 0
        const d0 = h0 > 0 ? (A[3 * k + r] - A[3 * (k - 1) + r]) / h0 : 0
        const d1 = h1 > 0 ? (A[3 * (k + 1) + r] - A[3 * k + r]) / h1 : 0
        let m = 0
        if (k === 0) m = d1
        else if (k === n - 1) m = d0
        else if (d0 * d1 > 0) {
          const w1 = 2 * h1 + h0, w2 = h1 + 2 * h0
          m = (w1 + w2) / (w1 / d0 + w2 / d1)
        }
        out[3 * k + r] = m
      }
    }
    return out
  }
  return { f, P, Y, Z, mP: tan(P), mY: tan(Y), mZ: tan(Z), n }
}

function mmss(t: number, tenths = false): string {
  if (!isFinite(t) || t < 0) t = 0
  const m = Math.floor(t / 60), s = t - 60 * m
  const ss = tenths ? s.toFixed(1).padStart(4, '0') : String(Math.floor(s)).padStart(2, '0')
  return `${String(m).padStart(2, '0')}:${ss}`
}

const RATES = ['0.25', '0.5', '1', '2'] as const
type Rate = typeof RATES[number]

/**
 * Synced video <-> 3D flythrough. Left = source video, right = the live 3D scene (driven via
 * viewportRef), a two-tier control bar below (USER 2026-10-10: "la barra de progreso casi no se
 * ve … agregale información de tiempo … debe ser absolutamente fluido").
 *
 * Playback contract:
 *  - The ENTIRE video is downloaded into a Blob first, so it can never buffer mid-play.
 *  - The cloud the path needs is PREFETCHED (pinned) while the video downloads, and the viewer
 *    switches to its flythrough render profile (discs, small sprites, 3 M points, no shadows,
 *    pixel ratio 1) — a saturated GPU used to freeze the cloud AND the video after a few seconds.
 *  - The camera moves at DISPLAY rate (requestAnimationFrame) on a clock anchored to the video's
 *    presented frames (requestVideoFrameCallback): no 30 Hz steps, no drift. The pose is a C1
 *    curve in time between keyframes (position, and the camera's axes, roll included).
 *  - Nothing in the hot loop touches React state: the bar, the playhead, the time and the frame
 *    readouts are written through refs. A frame that throws is logged and the loop goes on.
 */
export function SyncPlayer({ sessionId, viewportRef, onClose }: {
  sessionId: string
  viewportRef: React.RefObject<ViewportHandle>
  onClose: () => void
}) {
  const t = useT()
  const fmt = useFmt()
  const videoRef = useRef<HTMLVideoElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)
  const timeRef = useRef<HTMLSpanElement>(null)
  const infoRef = useRef<HTMLSpanElement>(null)
  const hoverRef = useRef<HTMLDivElement>(null)
  const [data, setData] = useState<Flythrough | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [dl, setDl] = useState(0)              // download progress 0..1 (-1 = indeterminate)
  const [canPlayThrough, setCanPlayThrough] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [rate, setRate] = useState<Rate>('1')
  const [cloudPct, setCloudPct] = useState<number | null>(0)   // prefetch of the path's cloud; null = done
  const [stalled, setStalled] = useState(false)

  const ready = !err && !!data && !!videoUrl && canPlayThrough

  const dataRef = useRef<Flythrough | null>(null)
  const pathRef = useRef<Path | null>(null)
  const fpsRef = useRef(30)
  const anchorRef = useRef<{ media: number; at: number } | null>(null)
  const draggingRef = useRef(false)
  const dragTRef = useRef(0)
  const wasPlayingRef = useRef(false)
  const pendingSeekRef = useRef<number | null>(null)
  const lastTRef = useRef(-1)
  const forceRef = useRef(true)
  const errLoggedRef = useRef(0)
  const outArr = useRef<number[]>(new Array(16).fill(0))
  const closedRef = useRef(false)

  // 1) Load poses + match the 3D camera FOV to the real camera + the flythrough render profile
  useEffect(() => {
    let alive = true
    closedRef.current = false
    fetch(`/api/sessions/${sessionId}/flythrough`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then((d: Flythrough) => {
        if (!alive) return
        if (!d.poses?.length) { setErr(t('flythrough.noPoses')); return }
        dataRef.current = d
        pathRef.current = buildPath(d)
        setData(d)
        const K = d.intrinsics
        let fovY = 60
        if (K && K.length >= 4 && K[1] > 0 && K[3] > 0) {
          fovY = 2 * Math.atan(K[3] / K[1]) * 180 / Math.PI
          viewportRef.current?.setCameraFov(fovY)
        }
        // the cloud along the whole path, fetched while the video downloads
        let last = 0
        viewportRef.current?.prefetchFlythrough(d.poses, fovY, frac => {
          const now = performance.now()
          if (frac >= 1 || now - last > 250) { last = now; if (alive) setCloudPct(frac >= 1 ? null : frac) }
        }, () => !alive).then(() => { if (alive) setCloudPct(null) }).catch(() => { if (alive) setCloudPct(null) })
      })
      .catch(() => { if (alive) setErr(t('flythrough.posesFailed')) })
    viewportRef.current?.setFlythroughActive(true)
    return () => { alive = false; closedRef.current = true; viewportRef.current?.setFlythroughActive(false) }
  }, [sessionId])

  // 2) Download the WHOLE video into a Blob (so playback never buffers)
  useEffect(() => {
    if (!data) return
    let alive = true
    let objUrl: string | null = null
    const ac = new AbortController()
    ;(async () => {
      try {
        const res = await fetch(data.video_url, { signal: ac.signal })
        if (!res.ok || !res.body) throw new Error(String(res.status))
        const total = Number(res.headers.get('Content-Length') || 0)
        setDl(total > 0 ? 0 : -1)
        const reader = res.body.getReader()
        const chunks: Uint8Array[] = []
        let recv = 0, lastUi = 0
        for (; ;) {
          const { done, value } = await reader.read()
          if (done) break
          if (value) {
            chunks.push(value); recv += value.length
            const now = performance.now()
            if (total > 0 && alive && now - lastUi > 150) { lastUi = now; setDl(recv / total) }
          }
        }
        if (!alive) return
        const blob = new Blob(chunks as BlobPart[], { type: res.headers.get('Content-Type') || 'video/mp4' })
        objUrl = URL.createObjectURL(blob)
        setDl(1)
        setVideoUrl(objUrl)
      } catch {
        if (alive) setErr(t('flythrough.videoFailed'))
      }
    })()
    return () => { alive = false; ac.abort(); if (objUrl) URL.revokeObjectURL(objUrl) }
  }, [data])

  // The pose at video time `sec` (C1 Hermite between the bracketing keyframes) → the viewport.
  // Returns the keyframe index (1-based) the time falls after.
  const applyPose = useCallback((sec: number): number => {
    const path = pathRef.current
    if (!path || !path.n) return 0
    const { f, P, Y, Z, mP, mY, mZ, n } = path
    let fr = sec * fpsRef.current
    if (fr < f[0]) fr = f[0]
    if (fr > f[n - 1]) fr = f[n - 1]
    let lo = 0, hi = n - 1
    if (n > 1) {
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (f[mid] <= fr) lo = mid; else hi = mid }
    } else hi = 0
    const h = f[hi] - f[lo]
    const s = h > 0 ? (fr - f[lo]) / h : 0
    const s2 = s * s, s3 = s2 * s
    const h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s, h01 = -2 * s3 + 3 * s2, h11 = s3 - s2
    const at = (A: Float64Array, M: Float64Array, r: number) =>
      h00 * A[3 * lo + r] + h10 * h * M[3 * lo + r] + h01 * A[3 * hi + r] + h11 * h * M[3 * hi + r]
    const px = at(P, mP, 0), py = at(P, mP, 1), pz = at(P, mP, 2)
    let zx = at(Z, mZ, 0), zy = at(Z, mZ, 1), zz = at(Z, mZ, 2)
    let yx = at(Y, mY, 0), yy = at(Y, mY, 1), yz = at(Y, mY, 2)
    // re-orthonormalise: z forward, x = y × z, y = z × x (OpenCV axes, right-handed)
    let l = Math.hypot(zx, zy, zz) || 1; zx /= l; zy /= l; zz /= l
    let xx = yy * zz - yz * zy, xy = yz * zx - yx * zz, xz = yx * zy - yy * zx
    l = Math.hypot(xx, xy, xz) || 1; xx /= l; xy /= l; xz /= l
    yx = zy * xz - zz * xy; yy = zz * xx - zx * xz; yz = zx * xy - zy * xx
    const o = outArr.current
    o[0] = xx; o[1] = yx; o[2] = zx; o[3] = px
    o[4] = xy; o[5] = yy; o[6] = zy; o[7] = py
    o[8] = xz; o[9] = yz; o[10] = zz; o[11] = pz
    o[12] = 0; o[13] = 0; o[14] = 0; o[15] = 1
    viewportRef.current?.setCameraToPose(o)
    return (s >= 0.5 ? hi : lo) + 1
  }, [viewportRef])

  // The bar, the playhead and the readouts — written through refs, never React state
  const paintBar = useCallback((sec: number, kf: number) => {
    const v = videoRef.current
    const d = dataRef.current
    const dur = v?.duration || 0
    const tn = dur > 0 ? Math.min(1, Math.max(0, sec / dur)) : 0
    trackRef.current?.style.setProperty('--p', String(tn))
    trackRef.current?.setAttribute('aria-valuenow', String(Math.round(tn * 1000) / 10))
    trackRef.current?.setAttribute('aria-valuetext', `${mmss(sec)} / ${mmss(dur)}`)
    if (timeRef.current) timeRef.current.textContent = `${mmss(sec, true)} / ${mmss(dur)}`
    if (infoRef.current && d) {
      const F = d.video_n_frames || Math.round(dur * fpsRef.current)
      const fr = Math.min(F, Math.max(1, Math.round(sec * fpsRef.current) + 1))
      infoRef.current.textContent = t('flythrough.frameInfo', { k: kf, n: d.n_frames || d.poses.length, f: fr, total: F })
    }
  }, [t])

  // 3) The display-rate loop: the camera moves every screen frame on a clock anchored to the
  //    presented video frames; the next frame is requested FIRST and the work runs guarded, so
  //    one bad frame can never stop the loop
  useEffect(() => {
    const v = videoRef.current
    if (!v || !ready) return
    const d = dataRef.current
    // the video's own average rate (its frames over its duration) — a variable-frame-rate phone
    // clip's nominal fps would drift the cloud out of sync along the walk
    fpsRef.current = (d?.video_n_frames && v.duration) ? d.video_n_frames / v.duration
      : (d?.fps && d.fps > 0) ? d.fps : 30
    let cancelled = false
    let raf = 0
    let vfc = 0
    const hasRVFC = typeof (v as any).requestVideoFrameCallback === 'function'
    const onVideoFrame = (_now: number, meta: any) => {
      if (cancelled) return
      vfc = (v as any).requestVideoFrameCallback(onVideoFrame)
      // anchored on the callback's frame time — the rAF timeline (expectedDisplayTime is a vsync
      // later and clamped the extrapolation to zero: the camera stepped at the video's rate)
      anchorRef.current = { media: meta.mediaTime, at: _now }
    }
    if (hasRVFC) vfc = (v as any).requestVideoFrameCallback(onVideoFrame)
    const tick = (now: number) => {
      if (cancelled) return
      raf = requestAnimationFrame(tick)
      try {
        const dur = v.duration || 0
        let sec: number
        const a = anchorRef.current
        if (draggingRef.current) sec = dragTRef.current
        else if (!v.paused && !v.seeking && a) {
          sec = a.media + Math.max(0, now - a.at) / 1000 * (v.playbackRate || 1)
          sec = Math.min(sec, a.media + 0.25)            // a stalled video holds the camera
        } else sec = v.currentTime
        if (dur > 0) sec = Math.min(dur, Math.max(0, sec))
        if (forceRef.current || Math.abs(sec - lastTRef.current) > 1e-4) {
          forceRef.current = false
          lastTRef.current = sec
          paintBar(sec, applyPose(sec))
        }
      } catch (e) {
        const n = ++errLoggedRef.current
        if (n <= 3 || n % 300 === 0) console.error('[SyncPlayer] frame failed (the loop goes on)', e)
      }
    }
    raf = requestAnimationFrame(tick)
    forceRef.current = true
    return () => {
      cancelled = true
      cancelAnimationFrame(raf)
      if (hasRVFC && typeof (v as any).cancelVideoFrameCallback === 'function') (v as any).cancelVideoFrameCallback(vfc)
    }
  }, [ready, applyPose, paintBar])

  // ── seeking: one seek in flight, the latest target pending (no seek storm while dragging)
  const seekTo = useCallback((sec: number, exact = false) => {
    const v = videoRef.current
    if (!v || !v.duration) return
    sec = Math.min(v.duration, Math.max(0, sec))
    anchorRef.current = null
    forceRef.current = true
    if (v.seeking && !exact) { pendingSeekRef.current = sec; return }
    pendingSeekRef.current = null
    const fast = (v as any).fastSeek
    if (!exact && typeof fast === 'function') fast.call(v, sec)
    else v.currentTime = sec
  }, [])
  const onSeeked = () => {
    anchorRef.current = null                     // a frame from before the seek is no anchor
    setStalled(false)
    const p = pendingSeekRef.current
    if (p != null) { pendingSeekRef.current = null; seekTo(p) }
    forceRef.current = true
  }

  const timeAt = (clientX: number) => {
    const el = trackRef.current, v = videoRef.current
    if (!el || !v?.duration) return 0
    const r = el.getBoundingClientRect()
    return Math.min(1, Math.max(0, (clientX - r.left) / Math.max(1, r.width))) * v.duration
  }
  const onTrackDown = (e: React.PointerEvent) => {
    const v = videoRef.current
    if (!ready || !v) return
    e.preventDefault()
    trackRef.current?.setPointerCapture(e.pointerId)
    draggingRef.current = true
    wasPlayingRef.current = !v.paused
    v.pause()
    dragTRef.current = timeAt(e.clientX)
    seekTo(dragTRef.current)
  }
  const onTrackMove = (e: React.PointerEvent) => {
    const sec = timeAt(e.clientX)
    if (hoverRef.current && videoRef.current?.duration) {
      hoverRef.current.textContent = mmss(sec, true)
      hoverRef.current.style.setProperty('--h', String(sec / videoRef.current.duration))
    }
    if (!draggingRef.current) return
    dragTRef.current = sec
    seekTo(sec)
  }
  const onTrackUp = (e: React.PointerEvent) => {
    if (!draggingRef.current) return
    draggingRef.current = false
    trackRef.current?.releasePointerCapture(e.pointerId)
    seekTo(dragTRef.current, true)
    if (wasPlayingRef.current) videoRef.current?.play().catch(() => { })
  }

  const toggle = useCallback(() => {
    const v = videoRef.current
    if (!v || !ready) return
    if (v.paused) { anchorRef.current = null; v.play().catch(() => { }) } else v.pause()
  }, [ready])
  const step = useCallback((frames: number) => {
    const v = videoRef.current
    if (!v || !ready) return
    v.pause()
    seekTo(v.currentTime + frames / (fpsRef.current || 30), true)
  }, [ready, seekTo])

  // keyboard: captured on the window so the app's own shortcuts (Home → reset camera) never
  // fight the player while it is open
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // the player's keys act only when the focus is the page, the 3D canvas or the player
      // itself — never over another widget (a chat input, a tree, a menu, a dialog) and never
      // with a modifier (Ctrl/Cmd+K, Ctrl+0 … stay the app's and the browser's)
      if (e.ctrlKey || e.metaKey || e.altKey) return
      const el = e.target as HTMLElement | null
      const ours = !el || el === document.body || el.tagName === 'CANVAS' || !!el.closest?.('.stac-sync')
      if (!ours || (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT'
          || el.isContentEditable))) return
      const v = videoRef.current
      let handled = true
      if (e.key === 'Escape') onClose()
      else if (!v || !ready) handled = e.key === 'Home' || e.key === 'End'
      else if (e.key === ' ' || e.key === 'k' || e.key === 'K') toggle()
      else if (e.key === 'ArrowLeft') seekTo(v.currentTime - (e.shiftKey ? 5 : 1), true)
      else if (e.key === 'ArrowRight') seekTo(v.currentTime + (e.shiftKey ? 5 : 1), true)
      else if (e.key === ',') step(-1)
      else if (e.key === '.') step(1)
      else if (e.key === 'Home') seekTo(0, true)
      else if (e.key === 'End') seekTo(v.duration, true)
      else if (/^[0-9]$/.test(e.key)) seekTo(v.duration * Number(e.key) / 10, true)
      else handled = false
      if (handled) { e.preventDefault(); e.stopImmediatePropagation() }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [ready, toggle, step, seekTo, onClose])

  useEffect(() => { if (videoRef.current) videoRef.current.playbackRate = Number(rate) }, [rate, videoUrl])

  const loadingLabel = !data ? t('flythrough.loadingScene') : !videoUrl ? (dl >= 0 ? t('flythrough.buffering', { pct: fmt.percent(dl) }) : t('flythrough.bufferingNoPct')) : t('flythrough.preparing')
  const ticks = (() => {
    if (!data || !data.frames?.length) return null
    const F = (data.video_n_frames && data.video_n_frames > 1) ? data.video_n_frames - 1 : data.frames[data.frames.length - 1] || 1
    return data.frames.map((fr, i) => <span key={i} className="stac-sync__tick" style={{ '--x': String(fr / F) } as React.CSSProperties} />)
  })()

  return (
    <div className="stac-sync">
      <div className="stac-sync__video">
        {err && <div className="stac-sync__error">{err}</div>}
        {videoUrl && (
          <video ref={videoRef} src={videoUrl} preload="auto" playsInline className="stac-sync__el"
            onCanPlayThrough={() => setCanPlayThrough(true)}
            onPlay={() => { anchorRef.current = null; setPlaying(true) }}
            onPause={() => { anchorRef.current = null; forceRef.current = true; setPlaying(false); setStalled(false) }}
            onCanPlay={() => setStalled(false)}
            onEnded={() => setPlaying(false)} onSeeked={onSeeked}
            onWaiting={() => setStalled(true)} onPlaying={() => setStalled(false)}
            onError={() => setErr(t('flythrough.videoDecodeFailed'))} />
        )}
        {!ready && !err && (
          <div className="stac-sync__loading"><Loader2 className="stac-spin" aria-hidden /><span>{loadingLabel}</span></div>
        )}
      </div>
      {!ready && !err && (
        <div className="stac-sync__loading stac-sync__loading--3d"><Loader2 className="stac-spin" aria-hidden /><span>{t('flythrough.loading3d')}</span></div>
      )}
      <div className="stac-sync__bar">
        <div ref={trackRef} className={`stac-sync__track ${ready ? '' : 'stac-sync__track--off'}`.trim()}
          role="slider" tabIndex={ready ? 0 : -1} aria-label={t('flythrough.seek')}
          aria-valuemin={0} aria-valuemax={100} aria-valuenow={0}
          style={{ '--p': '0', '--dl': String(Math.max(0, dl)) } as React.CSSProperties}
          onPointerDown={onTrackDown} onPointerMove={onTrackMove} onPointerUp={onTrackUp} onPointerCancel={onTrackUp}>
          <div className="stac-sync__rail">
            <div className="stac-sync__loaded" />
            <div className="stac-sync__fill" />
            {ticks}
          </div>
          <div className="stac-sync__handle" />
          <div ref={hoverRef} className="stac-sync__hover stac-mono" />
        </div>
        <div className="stac-sync__row">
          <IconButton size="lg" label={playing ? t('flythrough.pause') : t('flythrough.play')} shortcut="Space"
            icon={playing ? <Pause aria-hidden /> : <Play aria-hidden />} onClick={toggle} disabled={!ready} />
          <IconButton label={t('flythrough.stepBack')} shortcut="," icon={<SkipBack aria-hidden />} onClick={() => step(-1)} disabled={!ready} />
          <IconButton label={t('flythrough.stepForward')} shortcut="." icon={<SkipForward aria-hidden />} onClick={() => step(1)} disabled={!ready} />
          <span ref={timeRef} className="stac-sync__time stac-mono">00:00.0 / 00:00</span>
          <span ref={infoRef} className="stac-sync__info stac-mono" />
          {stalled && ready && <span className="stac-sync__chip"><Loader2 className="stac-spin" aria-hidden />{t('flythrough.stalled')}</span>}
          {cloudPct != null && data && (
            <span className="stac-sync__chip"><Loader2 className="stac-spin" aria-hidden />{t('flythrough.cloudLoading', { pct: fmt.percent(cloudPct) })}</span>
          )}
          <span className="stac-sync__spacer" />
          <SegmentedControl<Rate> size="sm" ariaLabel={t('flythrough.speed')} value={rate} onChange={setRate}
            options={RATES.map(r => ({ value: r, label: `${r}×` }))} />
          <IconButton label={t('flythrough.close')} shortcut="Esc" icon={<X aria-hidden />} onClick={onClose} />
        </div>
      </div>
    </div>
  )
}
