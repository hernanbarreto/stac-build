// STAC-Builder — immersive spatial AI assistant panel.
//
// Chat over the reconstruction: questions go to POST /api/spatial_qa (Phase 5
// orchestrator over the deterministic tools), and every answer's tool trace is
// replayed as ANIMATED geometry in the viewport (distances, volumes, angles,
// bounding boxes) so the user sees exactly HOW each number was measured. Also
// lets the user drop evaluation volumes into the scene to assess spaces
// (occupancy, free m³, whether an item fits).
//
// The assistant proposes/orchestrates; the tools measure. Answers are grounded
// in tool_measured geometry and traceable.
//
// Hernán Barreto - Ingerop IN3 Session IV - STAC

import { useCallback, useEffect, useRef, useState } from 'react'
import { Send, Box as BoxIcon, Trash2, Sparkles, Ruler, Loader2 } from 'lucide-react'
import type { ViewportHandle } from './Viewport'
import type { UserVolume } from './assistantViz'

interface TraceEntry {
    tool: string
    arguments: Record<string, unknown>
    result: Record<string, unknown>
}

interface Message {
    role: 'user' | 'assistant'
    text: string
    trace?: TraceEntry[]
    pending?: boolean
    note?: string          // what the pending bubble is waiting on
    error?: boolean
}

interface Props {
    sessionId: string | null
    viewport: React.RefObject<ViewportHandle | null>
    /** Model state changes (up | loading | busy | down) — drives the menubar icon. */
    onVlmStatus?: (status: 'up' | 'loading' | 'busy' | 'down') => void
}

const SUGGESTIONS = [
    'How many objects are in the scene?',
    'What is the clearance between the two nearest walls?',
    'Is the wall plumb?',
    'What defects have been reported?',
    'How much free space is in a 2×2×2 m box at the centre?',
]

const TOOL_LABELS: Record<string, string> = {
    get_distance: 'measured distance', get_clearance: 'measured clearance',
    get_object_size: 'measured size', get_object_volume: 'computed volume',
    get_position: 'located object', get_plumb: 'checked plumb', get_level: 'checked level',
    get_span: 'measured span', get_findings: 'read findings',
    get_alignment_health: 'checked alignment', get_onion_report: 'checked double-surface',
    list_objects: 'listed objects', count_objects: 'counted objects',
    evaluate_volume: 'evaluated volume', objects_in_volume: 'found objects in volume',
    fits_in_volume: 'checked fit', define_volume: 'defined volume',
    get_instance_history: 'read history',
    measure_between: 'measured between parts', get_extent: 'measured extent',
    get_session_info: 'read session info', describe_scene: 'looked at the scene',
    remember_note: 'saved a note', recall_notes: 'recalled notes',
    fits_through: 'checked passage', get_height_profile: 'height profile',
    get_flatness_report: 'checked flatness', get_my_position: 'located camera',
    get_distance_from_me: 'measured from camera',
}

function api(session: string | null, path: string, body: Record<string, unknown>) {
    return fetch(path, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: session, ...body }),
    })
}

export default function AssistantPanel({ sessionId, viewport, onVlmStatus }: Props) {
    const [messages, setMessages] = useState<Message[]>([])
    const [input, setInput] = useState('')
    const [busy, setBusy] = useState(false)
    const [volumes, setVolumes] = useState<UserVolume[]>([])
    const [showVolForm, setShowVolForm] = useState(false)
    const [vlmStatus, setVlmStatus] = useState<'up' | 'loading' | 'busy' | 'down' | null>(null)
    const [volForm, setVolForm] = useState({ name: 'Bay', cx: '0', cy: '0', cz: '0', w: '2', h: '2', d: '2' })
    const scrollRef = useRef<HTMLDivElement>(null)

    // Volumes: fetch, draw in the viewport, and tint by collision state.
    const refreshVolumes = useCallback(async () => {
        if (!sessionId) return
        try {
            const r = await api(sessionId, '/api/scene/volumes/list', {})
            const d = await r.json()
            if (!Array.isArray(d.volumes)) return
            setVolumes(d.volumes)
            for (const v of d.volumes as UserVolume[]) {
                viewport.current?.addUserVolume(v)
                try {
                    const er = await api(sessionId, '/api/scene/volumes/evaluate',
                        { volume_id: v.volume_id })
                    const ed = await er.json()
                    const occ = 1 - (typeof ed.free_fraction === 'number' ? ed.free_fraction : 1)
                    viewport.current?.setVolumeStatus(v.volume_id,
                        occ < 0.02 ? 'free' : occ < 0.12 ? 'touching' : 'colliding')
                } catch { /* keep default color */ }
            }
        } catch { /* ignore */ }
    }, [sessionId, viewport])

    // Load scene objects (for OBB-aware animations) + saved volumes on session change
    useEffect(() => {
        if (!sessionId) return
        let cancelled = false
        api(sessionId, '/api/scene/objects', {}).then((r) => r.json()).then((d) => {
            if (!cancelled && d.objects) viewport.current?.setAssistantObjects(d.objects)
        }).catch(() => { /* store may not exist yet */ })
        refreshVolumes()
        return () => { cancelled = true }
    }, [sessionId, viewport, refreshVolumes])

    useEffect(() => {
        scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
    }, [messages])

    // The reconstruction stages stop Qwen3-VL to free the GPU, so it is usually down
    // when the chat opens. Ask for its state and start it right away (the server
    // refuses while a pipeline is running), then poll until it serves — so the user
    // sees "loading" up front instead of discovering it after asking.
    useEffect(() => {
        let cancelled = false
        let warmed = false
        const tick = async () => {
            if (cancelled) return
            try {
                const r = await fetch(`/api/semantic/status?warmup=${warmed ? 'false' : 'true'}`)
                const d = await r.json()
                warmed = true
                if (cancelled) return
                setVlmStatus(d.status)
                onVlmStatus?.(d.status)
                // keep polling even when up — the model unloads during a
                // reconstruction and the icon/state must reflect it
                setTimeout(tick, d.status === 'up' ? 30000 : 8000)
            } catch {
                if (!cancelled) setTimeout(tick, 15000)
            }
        }
        tick()
        return () => { cancelled = true }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    const ask = useCallback(async (question: string) => {
        if (!question.trim() || busy) return
        // No session, or a session without segmentation, is fine: the backend
        // falls back to general chat and answers with whatever it has
        // (user 2026-08-28) — measurements just need a segmented session.
        setMessages((m) => [...m, { role: 'user', text: question },
            { role: 'assistant', text: '', pending: true }])
        setInput(''); setBusy(true)
        try {
            // The GPU is handed to the reconstruction, so Qwen3-VL is usually unloaded
            // when the chat opens. The server starts it and answers status=loading;
            // keep the pending bubble and wait for the weights instead of surfacing
            // the handover as an error the user has to act on.
            let d: Record<string, unknown>
            const deadline = Date.now() + 10 * 60_000
            for (;;) {
                const r = await api(sessionId, '/api/spatial_qa', { question })
                d = await r.json()
                if (r.ok) break
                if (r.status !== 503 || d.status !== 'loading' || Date.now() > deadline)
                    throw new Error(String(d.error || `HTTP ${r.status}`))
                setMessages((m) => replaceLast(m, {
                    role: 'assistant', text: '', pending: true,
                    note: 'Loading the model (Qwen3-VL)…',
                }))
                await new Promise((res) => setTimeout(res, 8000))
            }
            const trace: TraceEntry[] = (d.tool_trace as TraceEntry[]) || []
            if (trace.length) {
                // Refresh the OBB map right before animating: the instance store
                // may have been (re)built after this panel mounted (lazy rebuild
                // on the first question), and without OBBs the box/plumb/level
                // animations silently draw nothing.
                try {
                    const or = await api(sessionId, '/api/scene/objects', {})
                    const od = await or.json()
                    if (od.objects) viewport.current?.setAssistantObjects(od.objects)
                } catch { /* keep whatever objects we had */ }
            }
            viewport.current?.visualizeMeasurement(trace)
            // A volume was defined/edited during this answer → show it NOW
            // (user 2026-08-29: volumes appeared "later, who knows when").
            if (trace.some((t) => t.tool === 'define_volume')) await refreshVolumes()
            setMessages((m) => replaceLast(m, { role: 'assistant', text: String(d.answer || '(no answer)'), trace }))
        } catch (e) {
            setMessages((m) => replaceLast(m, {
                role: 'assistant', error: true,
                text: `Could not answer: ${e instanceof Error ? e.message : String(e)}`,
            }))
        } finally {
            setBusy(false)
        }
    }, [sessionId, busy, viewport, refreshVolumes])

    const addVolume = useCallback(async () => {
        if (!sessionId) return
        const center = [Number(volForm.cx), Number(volForm.cy), Number(volForm.cz)]
        const size = [Number(volForm.w), Number(volForm.h), Number(volForm.d)]
        if (size.some((s) => !(s > 0)) || center.some((c) => Number.isNaN(c))) return
        const r = await api(sessionId, '/api/scene/volumes/add', { name: volForm.name, center, size })
        const v = await r.json()
        if (v.volume_id != null) {
            setVolumes((prev) => [...prev, v])
            viewport.current?.addUserVolume(v)
            setShowVolForm(false)
            ask(`Evaluate the volume "${v.name}": what is inside and how much free space is there?`)
        }
    }, [sessionId, volForm, viewport, ask])

    const removeVolume = useCallback(async (id: number) => {
        if (!sessionId) return
        await api(sessionId, '/api/scene/volumes/delete', { volume_id: id })
        setVolumes((prev) => prev.filter((v) => v.volume_id !== id))
        viewport.current?.removeUserVolume(id)
    }, [sessionId, viewport])

    return (
        <div className="assistant-panel">
            {vlmStatus && vlmStatus !== 'up' && (
                <div className="assistant-status">
                    {vlmStatus === 'busy' ? (
                        <>⏸ Model unloaded — the GPU is busy reconstructing; the assistant
                            loads when it finishes.</>
                    ) : vlmStatus === 'down' ? (
                        <><Loader2 size={12} className="spin" /> Model unloaded — starting it now…</>
                    ) : (
                        <><Loader2 size={12} className="spin" /> Loading the model (Qwen3-VL)…</>
                    )}
                </div>
            )}
            <div className="assistant-scroll" ref={scrollRef}>
                {messages.length === 0 && (
                    <div className="assistant-empty">
                        <Sparkles size={22} />
                        <p>Ask anything. With a <b>segmented session</b> every figure is
                            measured by the geometry tools and <b>animated in 3D</b>;
                            without one the assistant still answers general questions.</p>
                        <div className="assistant-suggestions">
                            {SUGGESTIONS.map((s) => (
                                <button key={s} className="chip" onClick={() => ask(s)}>{s}</button>
                            ))}
                        </div>
                    </div>
                )}
                {messages.map((m, i) => (
                    <div key={i} className={`assistant-msg ${m.role} ${m.error ? 'error' : ''}`}>
                        {m.pending ? (
                            <span className="assistant-thinking">
                                <Loader2 size={14} className="spin" /> {m.note || 'thinking…'}
                            </span>
                        ) : (
                            <>
                                <div className="assistant-text">{m.text}</div>
                                {m.trace && m.trace.length > 0 && (
                                    <div className="assistant-trace">
                                        <Ruler size={12} />
                                        {m.trace.map((t, j) => (
                                            <span key={j} className="trace-pill" title={JSON.stringify(t.result)}>
                                                {TOOL_LABELS[t.tool] || t.tool}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                ))}
            </div>

            <div className="assistant-volumes">
                <div className="assistant-volumes-head">
                    <span><BoxIcon size={13} /> Evaluation volumes</span>
                    <button className="mini-btn" onClick={() => setShowVolForm((s) => !s)}>
                        {showVolForm ? 'Cancel' : '+ Add'}
                    </button>
                </div>
                {showVolForm && (
                    <div className="vol-form">
                        <input className="vol-name" value={volForm.name}
                            onChange={(e) => setVolForm({ ...volForm, name: e.target.value })} placeholder="Name" />
                        <div className="vol-row">
                            <label>center</label>
                            {(['cx', 'cy', 'cz'] as const).map((k) => (
                                <input key={k} value={volForm[k]} inputMode="decimal"
                                    onChange={(e) => setVolForm({ ...volForm, [k]: e.target.value })} />
                            ))}
                        </div>
                        <div className="vol-row">
                            <label>size</label>
                            {(['w', 'h', 'd'] as const).map((k) => (
                                <input key={k} value={volForm[k]} inputMode="decimal"
                                    onChange={(e) => setVolForm({ ...volForm, [k]: e.target.value })} />
                            ))}
                        </div>
                        <button className="vol-add" onClick={addVolume}>Place &amp; evaluate</button>
                    </div>
                )}
                {volumes.map((v) => (
                    <div key={v.volume_id} className="vol-item">
                        <button className="vol-focus" title="Frame in view"
                            onClick={() => viewport.current?.frameBox(
                                [v.center[0] - v.size[0] / 2, v.center[1] - v.size[1] / 2, v.center[2] - v.size[2] / 2],
                                [v.center[0] + v.size[0] / 2, v.center[1] + v.size[1] / 2, v.center[2] + v.size[2] / 2])}>
                            {v.name}
                        </button>
                        <span className="vol-dims">{v.size.map((s) => s.toFixed(1)).join('×')} m</span>
                        <button className="vol-eval" onClick={() =>
                            ask(`Evaluate volume ${v.volume_id}: objects inside and free space.`)}>eval</button>
                        <button className="vol-del" onClick={() => removeVolume(v.volume_id)}><Trash2 size={12} /></button>
                    </div>
                ))}
            </div>

            <form className="assistant-input" onSubmit={(e) => { e.preventDefault(); ask(input) }}>
                <input value={input} disabled={busy || vlmStatus !== 'up'}
                    placeholder={vlmStatus === 'up' ? 'Ask about the scene…'
                        : vlmStatus === 'loading' ? 'Loading the model…'
                        : vlmStatus === 'busy' ? 'GPU busy — model unloaded'
                        : 'Model unloaded — starting…'}
                    onChange={(e) => setInput(e.target.value)} />
                <button type="submit" disabled={busy || !input.trim() || vlmStatus !== 'up'}
                    aria-label="Send">
                    {busy ? <Loader2 size={16} className="spin" /> : <Send size={16} />}
                </button>
            </form>
        </div>
    )
}

function replaceLast(list: Message[], msg: Message): Message[] {
    const out = list.slice()
    out[out.length - 1] = msg
    return out
}
