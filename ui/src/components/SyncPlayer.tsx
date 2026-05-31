import { useEffect, useRef, useState } from 'react'
import { Play, Pause, X } from 'lucide-react'
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
 * spanning both. On playback the video time maps to a frame index → its ViPE
 * pose → the 3D camera, so the scene navigates in lockstep with the video.
 *
 * The viewport itself is shrunk to the right half by App (flythroughOpen state);
 * this overlay renders the left video column + the bottom control bar.
 */
export function SyncPlayer({ sessionId, viewportRef, onClose }: {
    sessionId: string
    viewportRef: React.RefObject<ViewportHandle>
    onClose: () => void
}) {
    const videoRef = useRef<HTMLVideoElement>(null)
    const [data, setData] = useState<Flythrough | null>(null)
    const [playing, setPlaying] = useState(false)
    const [progress, setProgress] = useState(0)
    const [err, setErr] = useState<string | null>(null)
    const rafRef = useRef<number | null>(null)

    useEffect(() => {
        fetch(`/api/sessions/${sessionId}/flythrough`)
            .then(r => r.ok ? r.json() : Promise.reject(r.status))
            .then((d: Flythrough) => {
                if (!d.poses?.length) { setErr('Sin poses (corré la reconstrucción con pose_source=vipe)'); return }
                setData(d)
            })
            .catch(() => setErr('No se pudieron cargar las poses del flythrough'))
        viewportRef.current?.setFlythroughActive(true)
        return () => { viewportRef.current?.setFlythroughActive(false) }
    }, [sessionId])

    // Drive the 3D camera from the video's current time, every animation frame.
    useEffect(() => {
        const v = videoRef.current
        if (!v || !data || !data.poses.length) return
        const tick = () => {
            const dur = v.duration || 1
            const tn = Math.min(1, Math.max(0, v.currentTime / dur))
            setProgress(tn)
            const idx = Math.min(data.poses.length - 1, Math.round(tn * (data.poses.length - 1)))
            viewportRef.current?.setCameraToPose(data.poses[idx])
            rafRef.current = requestAnimationFrame(tick)
        }
        rafRef.current = requestAnimationFrame(tick)
        return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
    }, [data])

    const toggle = () => {
        const v = videoRef.current
        if (!v) return
        if (v.paused) { v.play(); setPlaying(true) } else { v.pause(); setPlaying(false) }
    }
    const seek = (e: React.ChangeEvent<HTMLInputElement>) => {
        const v = videoRef.current
        if (!v || !v.duration) return
        v.currentTime = parseFloat(e.target.value) * v.duration
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
                        onEnded={() => setPlaying(false)}
                        playsInline
                        style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000' }}
                    />
                )}
            </div>
            {/* BOTTOM: thin shared control row spanning video + 3D */}
            <div className="sync-control-row">
                <button className="sync-btn" onClick={toggle} title={playing ? 'Pause' : 'Play'}>
                    {playing ? <Pause size={16} /> : <Play size={16} />}
                </button>
                <input
                    className="sync-seek"
                    type="range" min={0} max={1} step={0.001}
                    value={progress} onChange={seek}
                />
                <button className="sync-btn" onClick={onClose} title="Cerrar flythrough">
                    <X size={16} />
                </button>
            </div>
        </div>
    )
}
