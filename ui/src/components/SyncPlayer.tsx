import { useEffect, useRef, useState, useCallback } from 'react'
import * as THREE from 'three'
import { Play, Pause, X, Loader2 } from 'lucide-react'
import type { ViewportHandle } from './Viewport'

interface Flythrough {
    n_frames: number
    frames: number[]           // REAL video frame index of each keyframe pose (ascending, non-uniform)
    poses: number[][]          // per-keyframe c2w (16 floats, row-major), scaled to cloud frame
    intrinsics: number[] | null // [fx, fy, cx, cy] at reconstruction resolution
    fps: number | null          // source video fps (for time→frame mapping)
    video_n_frames: number | null // total frames in the source video
    video_url: string
}

/**
 * Synced video ↔ 3D flythrough. Left = source video, right = the live 3D scene
 * (driven via viewportRef), thin shared control row below.
 *
 * Playback contract (no stutter, ever):
 *  - The ENTIRE video is downloaded into a Blob up front (with a progress %),
 *    then served from an object URL → playback is 100% local, so it can NEVER
 *    buffer mid-play. Play is disabled until the video is fully downloaded AND
 *    the poses are loaded AND the element reports canplaythrough.
 *  - The 3D camera's vertical FOV is matched to the real camera (from the
 *    reconstruction intrinsics: fovY = 2·atan(cy/fy)), so the scene is framed
 *    exactly like the video — same zoom — instead of the viewport's default FOV.
 *  - Sync driver is requestVideoFrameCallback (frame-accurate, no polling, no
 *    work while paused); rAF fallback otherwise. The hot loop never touches React
 *    state (the seek slider is updated via a ref) — that re-render storm was the
 *    old jank. The camera pose is interpolated (lerp + slerp) between surrounding
 *    poses for a smooth glide; THREE scratch objects are pre-allocated.
 */
export function SyncPlayer({ sessionId, viewportRef, onClose }: {
    sessionId: string
    viewportRef: React.RefObject<ViewportHandle>
    onClose: () => void
}) {
    const videoRef = useRef<HTMLVideoElement>(null)
    const sliderRef = useRef<HTMLInputElement>(null)
    const [data, setData] = useState<Flythrough | null>(null)
    const [videoUrl, setVideoUrl] = useState<string | null>(null)
    const [dl, setDl] = useState(0)              // download progress 0..1 (-1 = indeterminate)
    const [canPlayThrough, setCanPlayThrough] = useState(false)
    const [playing, setPlaying] = useState(false)
    const [err, setErr] = useState<string | null>(null)

    // Everything is ready and playback is guaranteed smooth.
    const ready = !err && !!data && !!videoUrl && canPlayThrough

    // Hot-loop refs (never trigger renders).
    const dataRef = useRef<Flythrough | null>(null)
    const loopId = useRef<number | null>(null)
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

    // ── 1) Load poses + match the 3D camera FOV to the real camera ──
    useEffect(() => {
        let alive = true
        fetch(`/api/sessions/${sessionId}/flythrough`)
            .then(r => r.ok ? r.json() : Promise.reject(r.status))
            .then((d: Flythrough) => {
                if (!alive) return
                if (!d.poses?.length) { setErr('No poses (run reconstruction with pose_source=vipe)'); return }
                dataRef.current = d
                setData(d)
                const K = d.intrinsics
                if (K && K.length >= 4 && K[1] > 0 && K[3] > 0) {
                    // fovY = 2·atan(cy/fy); cy ≈ H/2 so this is the true vertical FOV.
                    const fovY = 2 * Math.atan(K[3] / K[1]) * 180 / Math.PI
                    viewportRef.current?.setCameraFov(fovY)
                }
            })
            .catch(() => { if (alive) setErr('Could not load flythrough poses') })
        viewportRef.current?.setFlythroughActive(true)
        return () => { alive = false; viewportRef.current?.setFlythroughActive(false) }
    }, [sessionId])

    // ── 2) Download the WHOLE video into a Blob (so playback never buffers) ──
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
                let recv = 0
                for (; ;) {
                    const { done, value } = await reader.read()
                    if (done) break
                    if (value) {
                        chunks.push(value)
                        recv += value.length
                        if (total > 0 && alive) setDl(recv / total)
                    }
                }
                if (!alive) return
                const blob = new Blob(chunks as BlobPart[], { type: res.headers.get('Content-Type') || 'video/mp4' })
                objUrl = URL.createObjectURL(blob)
                setDl(1)
                setVideoUrl(objUrl)
            } catch {
                if (alive) setErr('Could not download video')
            }
        })()
        return () => { alive = false; ac.abort(); if (objUrl) URL.revokeObjectURL(objUrl) }
    }, [data])

    // Map normalized VIDEO time tn∈[0,1] → interpolated camera pose, push it.
    //
    // Poses exist only at KEYFRAMES (d.frames[k] = the real video-frame index of
    // pose k), spaced non-uniformly. So we can't treat tn as a keyframe index —
    // that drifts out of sync wherever keyframes are sparse. Instead:
    //   1. tn → the real frame currently on screen:  targetFrame = tn·(total-1)
    //   2. find the two keyframes bracketing it:      frames[k] ≤ targetFrame ≤ frames[k+1]
    //   3. interpolate (lerp + slerp) by the position WITHIN that frame gap.
    // This keeps the camera glued to whatever frame the video shows, smoothly,
    // without needing a pose for every frame. Updates the slider imperatively
    // (no setState in the hot path).
    const applyPose = useCallback((tn: number) => {
        const d = dataRef.current
        if (!d || !d.poses.length) return
        tn = tn < 0 ? 0 : tn > 1 ? 1 : tn
        const s = sliderRef.current
        if (s && document.activeElement !== s) s.value = String(tn)

        const poses = d.poses
        const frames = d.frames
        const n = poses.length
        if (n === 1) { viewportRef.current?.setCameraToPose(poses[0]); return }

        // Total frames in the video → maps playback time to a real frame index.
        // Fall back to (last keyframe + 1) if the backend couldn't read it.
        const lastFrame = frames[n - 1]
        const totalFrames = (d.video_n_frames && d.video_n_frames > lastFrame)
            ? d.video_n_frames : lastFrame + 1
        const targetFrame = tn * (totalFrames - 1)

        // Before the first / after the last keyframe: clamp to the end pose.
        if (targetFrame <= frames[0]) { viewportRef.current?.setCameraToPose(poses[0]); return }
        if (targetFrame >= lastFrame) { viewportRef.current?.setCameraToPose(poses[n - 1]); return }

        // Binary search for k with frames[k] ≤ targetFrame < frames[k+1].
        let lo = 0, hi = n - 1
        while (hi - lo > 1) {
            const mid = (lo + hi) >> 1
            if (frames[mid] <= targetFrame) lo = mid; else hi = mid
        }
        const f0 = frames[lo], f1 = frames[lo + 1]
        const frac = f1 > f0 ? (targetFrame - f0) / (f1 - f0) : 0
        if (frac < 1e-4) { viewportRef.current?.setCameraToPose(poses[lo]); return }

        mA.current.fromArray(poses[lo]).transpose().decompose(pA.current, qA.current, scratch.current)
        mB.current.fromArray(poses[lo + 1]).transpose().decompose(pB.current, qB.current, scratch.current)
        pA.current.lerp(pB.current, frac)
        qA.current.slerp(qB.current, frac)
        outM.current.compose(pA.current, qA.current, ONE.current).transpose().toArray(outArr.current)
        viewportRef.current?.setCameraToPose(outArr.current)
    }, [viewportRef])

    // ── 3) Sync driver (only once fully ready) ──
    useEffect(() => {
        const v = videoRef.current
        if (!v || !ready) return
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
        applyPose(norm(v.currentTime || 0))   // place camera at the start frame

        return () => {
            cancelled = true
            if (loopId.current != null) {
                if (hasRVFC && typeof (v as any).cancelVideoFrameCallback === 'function')
                    (v as any).cancelVideoFrameCallback(loopId.current)
                else cancelAnimationFrame(loopId.current)
            }
        }
    }, [ready, applyPose])

    const toggle = () => {
        const v = videoRef.current
        if (!v || !ready) return
        if (v.paused) v.play().catch(() => { }); else v.pause()
    }
    const onSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
        const v = videoRef.current
        if (!v || !v.duration) return
        const tn = parseFloat(e.target.value)
        v.currentTime = tn * v.duration
        applyPose(tn)
    }

    const loadingLabel = !data ? 'Loading scene…'
        : !videoUrl ? (dl >= 0 ? `Buffering video… ${Math.round(dl * 100)}%` : 'Buffering video…')
            : 'Preparing…'

    return (
        <div className="sync-player-overlay">
            {/* LEFT: video (right half is the live 3D viewport, shrunk by App) */}
            <div className="sync-video-col">
                {err && <div className="sync-err">{err}</div>}
                {videoUrl && (
                    <video
                        ref={videoRef}
                        src={videoUrl}
                        preload="auto"
                        playsInline
                        onCanPlayThrough={() => setCanPlayThrough(true)}
                        onPlay={() => setPlaying(true)}
                        onPause={() => setPlaying(false)}
                        onEnded={() => setPlaying(false)}
                        style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000' }}
                    />
                )}
                {!ready && !err && (
                    <div className="sync-loading">
                        <Loader2 size={30} className="sync-spin" />
                        <span className="sync-loading-label">{loadingLabel}</span>
                    </div>
                )}
            </div>
            {/* 3D-side indicator (right half) mirrors the loading state. */}
            {!ready && !err && (
                <div className="sync-loading sync-loading-3d">
                    <Loader2 size={30} className="sync-spin" />
                    <span className="sync-loading-label">Loading 3D scene…</span>
                </div>
            )}
            {/* BOTTOM: thin shared control row. Controls disabled until fully buffered. */}
            <div className="sync-control-row">
                <button className="sync-btn" onClick={toggle} disabled={!ready}
                    title={playing ? 'Pause' : 'Play'}>
                    {playing ? <Pause size={16} /> : <Play size={16} />}
                </button>
                <input
                    ref={sliderRef}
                    className="sync-seek"
                    type="range" min={0} max={1} step={0.001}
                    defaultValue={0} onChange={onSeek} disabled={!ready}
                />
                <button className="sync-btn" onClick={onClose} title="Close flythrough">
                    <X size={16} />
                </button>
            </div>
        </div>
    )
}
