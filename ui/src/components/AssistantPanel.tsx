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
import { Send, Box as BoxIcon, Trash2, Sparkles, Ruler, Loader2, Crosshair, Plus, Pause } from 'lucide-react'
import type { ViewportHandle } from './Viewport'
import type { UserVolume } from './assistantViz'
import { Button } from './ui/Button'
import { IconButton } from './ui/IconButton'
import { Input, NumberInput } from './ui/Field'
import { Banner } from './ui/Banner'
import { Badge } from './ui/Badge'
import { Section, Row, Stack } from './ui/Panel'
import { Tooltip } from './ui/Tooltip'
import { useFmt, useT } from '../i18n'

interface TraceEntry { tool: string; arguments: Record<string, unknown>; result: Record<string, unknown> }

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
  /** Model state changes (up | loading | busy | down) — drives the header icon. */
  onVlmStatus?: (status: 'up' | 'loading' | 'busy' | 'down') => void
}

const SUGGESTION_KEYS = ['count', 'clearance', 'plumb', 'defects', 'freeSpace'] as const

function api(session: string | null, path: string, body: Record<string, unknown>) {
  return fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: session, ...body }) })
}

export default function AssistantPanel({ sessionId, viewport, onVlmStatus }: Props) {
  const t = useT()
  const fmt = useFmt()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [volumes, setVolumes] = useState<UserVolume[]>([])
  const [showVolForm, setShowVolForm] = useState(false)
  const [vlmStatus, setVlmStatus] = useState<'up' | 'loading' | 'busy' | 'down' | null>(null)
  const [volForm, setVolForm] = useState<{ name: string; cx: number | ''; cy: number | ''; cz: number | ''; w: number | ''; h: number | ''; d: number | '' }>({ name: 'Bay', cx: 0, cy: 0, cz: 0, w: 2, h: 2, d: 2 })
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
          const er = await api(sessionId, '/api/scene/volumes/evaluate', { volume_id: v.volume_id })
          const ed = await er.json()
          const occ = 1 - (typeof ed.free_fraction === 'number' ? ed.free_fraction : 1)
          viewport.current?.setVolumeStatus(v.volume_id, occ < 0.02 ? 'free' : occ < 0.12 ? 'touching' : 'colliding')
        } catch { /* keep default colour */ }
      }
    } catch { /* ignore */ }
  }, [sessionId, viewport])

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    api(sessionId, '/api/scene/objects', {}).then(r => r.json()).then(d => {
      if (!cancelled && d.objects) viewport.current?.setAssistantObjects(d.objects)
    }).catch(() => { /* store may not exist yet */ })
    refreshVolumes()
    return () => { cancelled = true }
  }, [sessionId, viewport, refreshVolumes])

  useEffect(() => { scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' }) }, [messages])

  // The reconstruction stages stop Qwen3-VL to free the GPU, so it is usually down
  // when the chat opens. Ask for its state and start it right away (the server
  // refuses while a pipeline is running), then poll until it serves.
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
        setTimeout(tick, d.status === 'up' ? 30000 : 8000)
      } catch {
        if (!cancelled) setTimeout(tick, 15000)
      }
    }
    tick()
    return () => { cancelled = true }
     
  }, [])

  const ask = useCallback(async (question: string) => {
    if (!question.trim() || busy) return
    setMessages(m => [...m, { role: 'user', text: question }, { role: 'assistant', text: '', pending: true }])
    setInput(''); setBusy(true)
    try {
      let d: Record<string, unknown>
      const deadline = Date.now() + 10 * 60_000
      for (;;) {
        const r = await api(sessionId, '/api/spatial_qa', { question })
        d = await r.json()
        if (r.ok) break
        if (r.status !== 503 || d.status !== 'loading' || Date.now() > deadline) throw new Error(String(d.error || `HTTP ${r.status}`))
        setMessages(m => replaceLast(m, { role: 'assistant', text: '', pending: true, note: t('assistant.modelLoading') }))
        await new Promise(res => setTimeout(res, 8000))
      }
      const trace: TraceEntry[] = (d.tool_trace as TraceEntry[]) || []
      if (trace.length) {
        try {
          const or = await api(sessionId, '/api/scene/objects', {})
          const od = await or.json()
          if (od.objects) viewport.current?.setAssistantObjects(od.objects)
        } catch { /* keep whatever objects we had */ }
      }
      viewport.current?.visualizeMeasurement(trace)
      if (trace.some(x => x.tool === 'define_volume')) await refreshVolumes()
      setMessages(m => replaceLast(m, { role: 'assistant', text: String(d.answer || t('assistant.noAnswer')), trace }))
    } catch (e) {
      setMessages(m => replaceLast(m, { role: 'assistant', error: true, text: t('assistant.couldNotAnswer', { detail: e instanceof Error ? e.message : String(e) }) }))
    } finally {
      setBusy(false)
    }
  }, [sessionId, busy, viewport, refreshVolumes, t])

  const addVolume = useCallback(async () => {
    if (!sessionId) return
    const center = [Number(volForm.cx), Number(volForm.cy), Number(volForm.cz)]
    const size = [Number(volForm.w), Number(volForm.h), Number(volForm.d)]
    if (size.some(s => !(s > 0)) || center.some(c => Number.isNaN(c))) return
    const r = await api(sessionId, '/api/scene/volumes/add', { name: volForm.name, center, size })
    const v = await r.json()
    if (v.volume_id != null) {
      setVolumes(prev => [...prev, v])
      viewport.current?.addUserVolume(v)
      setShowVolForm(false)
      ask(t('assistant.evaluateVolumeQuestion', { name: v.name }))
    }
  }, [sessionId, volForm, viewport, ask, t])

  const removeVolume = useCallback(async (id: number) => {
    if (!sessionId) return
    await api(sessionId, '/api/scene/volumes/delete', { volume_id: id })
    setVolumes(prev => prev.filter(v => v.volume_id !== id))
    viewport.current?.removeUserVolume(id)
  }, [sessionId, viewport])

  const inputPlaceholder = vlmStatus === 'up' ? t('assistant.placeholder') : vlmStatus === 'loading' ? t('assistant.modelLoading') : vlmStatus === 'busy' ? t('assistant.modelBusy') : t('assistant.modelStarting')

  return (
    <div className="stac-assistant">
      {vlmStatus && vlmStatus !== 'up' && (
        <Banner tone="info" compact icon={vlmStatus === 'busy' ? <Pause aria-hidden /> : <Loader2 className="stac-spin" aria-hidden />}>
          {vlmStatus === 'busy' ? t('assistant.busyBanner') : vlmStatus === 'down' ? t('assistant.downBanner') : t('assistant.loadingBanner')}
        </Banner>
      )}
      <div className="stac-assistant__scroll" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="stac-assistant__empty">
            <Sparkles aria-hidden />
            <p>{t('assistant.emptyIntro')}</p>
            <div className="stac-assistant__suggestions">
              {SUGGESTION_KEYS.map(k => <button key={k} type="button" className="stac-assistant__chip" onClick={() => ask(t(`assistant.suggestions.${k}`))}>{t(`assistant.suggestions.${k}`)}</button>)}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`stac-assistant__msg stac-assistant__msg--${m.role} ${m.error ? 'stac-assistant__msg--error' : ''}`.trim()}>
            {m.pending ? (
              <span className="stac-assistant__thinking"><Loader2 className="stac-spin" aria-hidden /> {m.note || t('assistant.thinking')}</span>
            ) : (
              <>
                <div className="stac-assistant__text stac-selectable">{m.text}</div>
                {m.trace && m.trace.length > 0 && (
                  <div className="stac-assistant__trace">
                    <Ruler aria-hidden />
                    {m.trace.map((tr, j) => (
                      <Tooltip key={j} label={<span className="stac-mono">{JSON.stringify(tr.result).slice(0, 240)}</span>}>
                        <span><Badge tone="measure" size="sm">{t.has(`toolLabel.${tr.tool}`) ? t(`toolLabel.${tr.tool}`) : tr.tool}</Badge></span>
                      </Tooltip>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        ))}
      </div>

      <Section title={<Row><BoxIcon aria-hidden className="stac-assistant__icon" />{t('assistant.volumes')}</Row>} className="stac-assistant__volumes"
        actions={<Button size="sm" variant="ghost" icon={<Plus aria-hidden />} onClick={() => setShowVolForm(s => !s)}>{showVolForm ? t('common.cancel') : t('assistant.addVolume')}</Button>}>
        {showVolForm && (
          <Stack gap={2} className="stac-assistant__volform">
            <Input size="sm" value={volForm.name} onChange={e => setVolForm({ ...volForm, name: e.target.value })} placeholder={t('assistant.volumeName')} aria-label={t('assistant.volumeName')} />
            <Row><span className="stac-assistant__vollabel">{t('assistant.center')}</span>
              {(['cx', 'cy', 'cz'] as const).map(k => <NumberInput key={k} size="sm" unit="m" value={volForm[k]} onChange={v => setVolForm({ ...volForm, [k]: v })} aria-label={`${t('assistant.center')} ${k.slice(1).toUpperCase()}`} />)}
            </Row>
            <Row><span className="stac-assistant__vollabel">{t('assistant.size')}</span>
              {(['w', 'h', 'd'] as const).map(k => <NumberInput key={k} size="sm" unit="m" value={volForm[k]} onChange={v => setVolForm({ ...volForm, [k]: v })} aria-label={`${t('assistant.size')} ${k.toUpperCase()}`} />)}
            </Row>
            <Button variant="primary" size="sm" icon={<BoxIcon aria-hidden />} onClick={addVolume}>{t('assistant.placeEvaluate')}</Button>
          </Stack>
        )}
        {volumes.map(v => (
          <Row key={v.volume_id} className="stac-assistant__vol">
            <button type="button" className="stac-assistant__volname" title={t('assistant.frameVolume')}
              onClick={() => viewport.current?.frameBox([v.center[0] - v.size[0] / 2, v.center[1] - v.size[1] / 2, v.center[2] - v.size[2] / 2], [v.center[0] + v.size[0] / 2, v.center[1] + v.size[1] / 2, v.center[2] + v.size[2] / 2])}>
              <Crosshair aria-hidden />{v.name}
            </button>
            <span className="stac-assistant__voldims stac-mono">{`${v.size.map(s => fmt.number(s, 1)).join(' x ')} m`}</span>
            <Button size="sm" variant="ghost" onClick={() => ask(t('assistant.evaluateVolumeById', { id: v.volume_id }))}>{t('assistant.evaluate')}</Button>
            <IconButton size="sm" variant="danger" label={t('common.delete')} icon={<Trash2 aria-hidden />} onClick={() => removeVolume(v.volume_id)} />
          </Row>
        ))}
      </Section>

      <form className="stac-assistant__input" onSubmit={e => { e.preventDefault(); ask(input) }}>
        <Input value={input} disabled={busy || vlmStatus !== 'up'} placeholder={inputPlaceholder} aria-label={t('assistant.placeholder')} onChange={e => setInput(e.target.value)} />
        <IconButton type="submit" variant="primary" tone="measure" label={t('assistant.send')} icon={<Send aria-hidden />} loading={busy} disabled={!input.trim() || vlmStatus !== 'up'} />
      </form>
    </div>
  )
}

function replaceLast(list: Message[], msg: Message): Message[] {
  const out = list.slice()
  out[out.length - 1] = msg
  return out
}
