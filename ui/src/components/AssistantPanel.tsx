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
    error?: boolean
}

interface Props {
    sessionId: string | null
    viewport: React.RefObject<ViewportHandle | null>
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
}

function api(session: string | null, path: string, body: Record<string, unknown>) {
    return fetch(path, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: session, ...body }),
    })
}

export default function AssistantPanel({ sessionId, viewport }: Props) {
    const [messages, setMessages] = useState<Message[]>([])
    const [input, setInput] = useState('')
    const [busy, setBusy] = useState(false)
    const [volumes, setVolumes] = useState<UserVolume[]>([])
    const [showVolForm, setShowVolForm] = useState(false)
    const [volForm, setVolForm] = useState({ name: 'Bay', cx: '0', cy: '0', cz: '0', w: '2', h: '2', d: '2' })
    const scrollRef = useRef<HTMLDivElement>(null)

    // Load scene objects (for OBB-aware animations) + saved volumes on session change
    useEffect(() => {
        if (!sessionId) return
        let cancelled = false
        api(sessionId, '/api/scene/objects', {}).then((r) => r.json()).then((d) => {
            if (!cancelled && d.objects) viewport.current?.setAssistantObjects(d.objects)
        }).catch(() => { /* store may not exist yet */ })
        api(sessionId, '/api/scene/volumes/list', {}).then((r) => r.json()).then((d) => {
            if (cancelled || !d.volumes) return
            setVolumes(d.volumes)
            d.volumes.forEach((v: UserVolume) => viewport.current?.addUserVolume(v))
        }).catch(() => { /* ignore */ })
        return () => { cancelled = true }
    }, [sessionId, viewport])

    useEffect(() => {
        scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
    }, [messages])

    const ask = useCallback(async (question: string) => {
        if (!question.trim() || busy) return
        if (!sessionId) {
            setMessages((m) => [...m, { role: 'assistant', text: 'Load a reconstructed session first.', error: true }])
            return
        }
        setMessages((m) => [...m, { role: 'user', text: question },
            { role: 'assistant', text: '', pending: true }])
        setInput(''); setBusy(true)
        try {
            const r = await api(sessionId, '/api/spatial_qa', { question })
            const d = await r.json()
            if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`)
            const trace: TraceEntry[] = d.tool_trace || []
            viewport.current?.visualizeMeasurement(trace)
            setMessages((m) => replaceLast(m, { role: 'assistant', text: d.answer || '(no answer)', trace }))
        } catch (e) {
            setMessages((m) => replaceLast(m, {
                role: 'assistant', error: true,
                text: `Could not answer: ${e instanceof Error ? e.message : String(e)}`,
            }))
        } finally {
            setBusy(false)
        }
    }, [sessionId, busy, viewport])

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
            <div className="assistant-scroll" ref={scrollRef}>
                {messages.length === 0 && (
                    <div className="assistant-empty">
                        <Sparkles size={22} />
                        <p>Ask about the reconstruction. Every answer is measured by the
                            geometry tools and <b>animated in 3D</b> so you see how.</p>
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
                            <span className="assistant-thinking"><Loader2 size={14} className="spin" /> measuring…</span>
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
                <input value={input} disabled={busy} placeholder="Ask about the scene…"
                    onChange={(e) => setInput(e.target.value)} />
                <button type="submit" disabled={busy || !input.trim()} aria-label="Send">
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
