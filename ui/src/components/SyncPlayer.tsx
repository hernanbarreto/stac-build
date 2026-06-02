import { useEffect, useRef, useState, useCallback } from 'react'
import * as THREE from 'three'
import { Play, Pause, X, Loader2 } from 'lucide-react'
import type { ViewportHandle } from './Viewport'

interface Flythrough {
    n_frames: number
    frames: number[]
    poses: number[][]      // per-frame c2w (16 floats, row-major), scaled to cloud frame
    video_url: string
}

/**
 * Synced video ↔ 3D flythrough. Left = source video, right = the live 3D scene
 * (driven via viewportRef), thin shared control row below (play/pause + seek)
 * spanning both. On playback the video time maps to a frame index → its pose →
 * the 3D camera, so the scene navigates in lockstep with the video.
 *
 * Smoothness/robustness design (was stuttering before):
 *  - Sync driver is requestVideoFrameCallback (rVFC): fires once per *presented*
 *    video frame with an exact mediaTime → frame-accurate AV↔3D sync, no polling,
 *    and ZERO work while paused/stalled. Falls back to rAF where rVFC is absent.
 *  - The hot loop does NOT touch React state (that re-render storm was the jank).
 *    The seek slider is updated imperatively via a DOM ref; React state is only
 *    used for play/pause + buffering, which change rarely.
 *  - The camera pose is interpolated (position lerp + rotation slerp) between the
 *    two surrounding poses, so the camera glides smoothly even when poses are
 *    sparse (keyframes / stride 2). All THREE objects are pre-allocated.
 *  - Play state + buffering are derived from real <video> events, so the UI never
 *    desyncs from the element and a network stall shows a spinner instead of a
 *    silent freeze.
 */
export function SyncPlayer({ sessionId, viewportRef, onClose }: {
    sessionId: string
    viewportRef: React.RefObject<ViewportHandle>
    onClose: () => void
}) {
    const videoRef = useRef<HTMLVideoElement>(null)
    const sliderRef = useRef<HTMLInputElement>(null)
    const [data, setData] = useState<Flythrough | null>(null)
    const [playing, setPlaying] = useState(false)
    const [buffering, setBuffering] = useState(false)
    const [ready, setReady] = useState(false)   // video has enough data to play
    const [err, setErr] = useState<string | null>(null)

    // Initial load: poses still fetching, or video not yet playable. Distinct from
    // `buffering`, which is a mid-playback stall.
    const loading = !err && (!data || !ready)

    // Hot-loop refs (mutated every frame, never trigger React renders).
    const dataRef = useRef<Flythrough | null>(null)
    const loopId = useRef<number | null>(null)
    // Pre-allocated THREE scratch (no per-frame GC churn).
    const mA = useRef(new THREE.Matrix4())
    const mB = useRef(new THREE.Matrix4())
    const pA = useRef(new THREE.Vector3())
    const pB = useRef(new THREE.Vector3())
    const qA = useRef(new THREE.Quaternion())
    const qB = useRef(new THREE.Quaternion())
    const scratch = useRef(new THREE.Vector3())
    const ONE = useRef(new THREE.Vector3(1, 1, 1))
    const outM = useRef(new THREE.Matrix4())
    const outArr = useRef<number[]>(new Array(16).fill(0))

    useEffect(() => {
        fetch(`/api/sessions/${sessionId}/flythrough`)
            .then(r => r.ok ? r.json() : Promise.reject(r.status))
            .then((d: Flythrough) => {
                if (!d.poses?.length) { setErr('No poses (run reconstruction with pose_source=vipe)'); return }
                dataRef.current = d
                setData(d)
            })
            .catch(() => setErr('Could not load flythrough poses'))
        viewportRef.current?.setFlythroughActive(true)
        return () => { viewportRef.current?.setFlythroughActive(false) }
    }, [sessionId])

    // Map a normalized time tn∈[0,1] → interpolated camera pose, and push it.
    // Updates the slider imperatively (no setState in the hot path).
    const applyPose = useCallback((tn: number) => {
        const d = dataRef.current
        if (!d || !d.poses.length) return
        tn = tn < 0 ? 0 : tn > 1 ? 1 : tn

        // Imperative slider update — skip while the user is dragging it.
        const s = sliderRef.current
        if (s && document.activeElement !== s) s.value = String(tn)

        const n = d.poses.length
        if (n === 1) { viewportRef.current?.setCameraToPose(d.poses[0]); return }

        const f = tn * (n - 1)
        const i0 = Math.min(n - 1, Math.floor(f))
        const i1 = Math.min(n - 1, i0 + 1)
        const frac = f - i0

        if (i0 === i1 || frac < 1e-4) {
            viewportRef.current?.setCameraToPose(d.poses[i0])
            return
        }

        // Interpolate in the raw c2w space (poses are row-major → fromArray reads
        // column-major, so transpose to recover the real matrix). lerp position,
        // slerp rotation, recompose, then transpose back to row-major for the
        // viewport (which applies its own transpose + CV→GL + floor transform).
        mA.current.fromArray(d.poses[i0]).transpose().decompose(pA.current, qA.current, scratch.current)
        mB.current.fromArray(d.poses[i1]).transpose().decompose(pB.current, qB.current, scratch.current)
        pA.current.lerp(pB.current, frac)
        qA.current.slerp(qB.current, frac)
        outM.current.compose(pA.current, qA.current, ONE.current).transpose().toArray(outArr.current)
        viewportRef.current?.setCameraToPose(outArr.current)
    }, [viewportRef])

    // Sync driver: rVFC (preferred) or rAF fallback. Re-created when data loads.
    useEffect(() => {
        const v = videoRef.current
        if (!v || !data) return
        const hasRVFC = typeof (v as any).requestVideoFrameCallback === 'function'
        let cancelled = false

        const norm = (t: number) => t / (v.duration || 1)

        if (hasRVFC) {
            const onFrame = (_now: number, meta: any) => {
                if (cancelled) return
                applyPose(norm(meta.mediaTime))
                loopId.current = (v as any).requestVideoFrameCallback(onFrame)
            }
            loopId.current = (v as any).requestVideoFrameCallback(onFrame)
        } else {
            const tick = () => {
                if (cancelled) return
                applyPose(norm(v.currentTime))
                loopId.current = requestAnimationFrame(tick)
            }
            loopId.current = requestAnimationFrame(tick)
        }

        // Show frame 0 immediately, before any playback.
        applyPose(norm(v.currentTime || 0))

        return () => {
            cancelled = true
            if (loopId.current != null) {
                if (hasRVFC && typeof (v as any).cancelVideoFrameCallback === 'function')
                    (v as any).cancelVideoFrameCallback(loopId.current)
                else cancelAnimationFrame(loopId.current)
            }
        }
    }, [data, applyPose])

    const toggle = () => {
        const v = videoRef.current
        if (!v) return
        if (v.paused) v.play().catch(() => { }); else v.pause()
    }
    const onSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
        const v = videoRef.current
        if (!v || !v.duration) return
        const tn = parseFloat(e.target.value)
        v.currentTime = tn * v.duration
        applyPose(tn)   // update camera even while paused (rVFC also fires post-seek)
    }

    return (
        <div className="sync-player-overlay">
            {/* LEFT: video (right half is the live 3D viewport, shrunk by App) */}
            <div className="sync-video-col">
                {err && <div className="sync-err">{err}</div>}
                {data && (
                    <video
                        ref={videoRef}
                        src={data.video_url}
                        preload="auto"
                        playsInline
                        onLoadedData={() => setReady(true)}
                        onCanPlay={() => { setReady(true); setBuffering(false) }}
                        onPlay={() => setPlaying(true)}
                        onPlaying={() => { setPlaying(true); setBuffering(false) }}
                        onPause={() => setPlaying(false)}
                        onWaiting={() => setBuffering(true)}
                        onStalled={() => setBuffering(true)}
                        onSeeking={() => setBuffering(true)}
                        onSeeked={() => setBuffering(false)}
                        onEnded={() => setPlaying(false)}
                        style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000' }}
                    />
                )}
                {/* Video-side indicator: initial load OR mid-playback buffering. */}
                {(loading || buffering) && (
                    <div className="sync-loading">
                        <Loader2 size={30} className="sync-spin" />
                        <span className="sync-loading-label">
                            {loading ? (data ? 'Loading video…' : 'Loading scene…') : 'Buffering…'}
                        </span>
                    </div>
                )}
            </div>
            {/* 3D-side indicator (right half): mirror the loading/buffering state so
                the scene zone also shows it's not ready yet. */}
            {(loading || buffering) && (
                <div className="sync-loading sync-loading-3d">
                    <Loader2 size={30} className="sync-spin" />
                    <span className="sync-loading-label">
                        {loading ? 'Loading 3D scene…' : 'Buffering…'}
                    </span>
                </div>
            )}
            {/* BOTTOM: thin shared control row spanning video + 3D */}
            <div className="sync-control-row">
                <button className="sync-btn" onClick={toggle} title={playing ? 'Pause' : 'Play'}>
                    {playing ? <Pause size={16} /> : <Play size={16} />}
                </button>
                <input
                    ref={sliderRef}
                    className="sync-seek"
                    type="range" min={0} max={1} step={0.001}
                    defaultValue={0} onChange={onSeek}
                />
                <button className="sync-btn" onClick={onClose} title="Cerrar flythrough">
                    <X size={16} />
                </button>
            </div>
        </div>
    )
}
