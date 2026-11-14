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
import { Send, Box as BoxIcon, Trash2, Sparkles, Ruler, Loader2, Crosshair, Plus, Pause, Mic, Square, Volume2, VolumeX } from 'lucide-react'
import type { ViewportHandle } from './Viewport'
import type { UserVolume } from './assistantViz'
import { Button } from './ui/Button'
import { IconButton } from './ui/IconButton'
import { Input, NumberInput } from './ui/Field'
import { Banner } from './ui/Banner'
import { Badge } from './ui/Badge'
import { Section, Row, Stack } from './ui/Panel'
import { Tooltip } from './ui/Tooltip'
import { useFmt, useI18n, useT } from '../i18n'
import { useVoice, voiceSupported, type VoiceLang } from '../voice/useVoice'
import { registerCommands, runActions, type AnswerTexts, type Executors, type UiAction } from '../commands/registry'

interface TraceEntry { tool: string; arguments: Record<string, unknown>; result: Record<string, unknown> }

interface Message {
  role: 'user' | 'assistant'
  text: string
  trace?: TraceEntry[]
  pending?: boolean
  note?: string          // what the pending bubble is waiting on
  error?: boolean
  /** figures / tools of the answer the server's provenance check could not trace to a tool
   *  call of this turn, after asking the model again — shown under the answer, never hidden */
  unverified?: string[]
}

interface Props {
  sessionId: string | null
  /** the scan the viewer shows ("date/source") — the chat answers about THIS scan's store
   *  (USER 2026-11-13: with pccr 3108 open it answered from pccr 2408) */
  scanKey?: string | null
  viewport: React.RefObject<ViewportHandle | null>
  /** Model state changes (up | loading | busy | down) — drives the header icon. */
  onVlmStatus?: (status: 'up' | 'loading' | 'busy' | 'down') => void
  /** The objects an answer is about, in the order its tools named them — the viewer selects and
   *  flies to them as a click would (USER 2026-10-09). */
  onFocusObjects?: (ids: number[]) => void
  /** a volume was added or removed here — the app's list (Instances) follows */
  onVolumesChanged?: () => void
  /** an action on the viewer one of the answer's tools returned (focus, show/hide, OBB, view,
   *  orbit, travel, colour, clear) — USER 2026-10-09: the chat drives the viewer */
  onViewerAction?: (action: Record<string, unknown>) => void
  /** who is signed in: STAC remembers each user's name (USER 2026-11-14: "debe preguntarnos
   *  primero el nombre, y hablarnos usando nuestro nombre") */
  userKey?: string | null
  /** what the interface says of itself, sent with every question (connected, session, scan,
   *  the sessions' names …) — the model needs it to operate the application */
  getUiState?: () => Record<string, unknown>
}

// the object ids the answer's tools were called on (get_position(id), measure_between(id1, id2),
// fits_through(opening_id) …), first mention first
function objectsOfTrace(trace: TraceEntry[]): number[] {
  const ids: number[] = []
  for (const e of trace) {
    for (const k of ['id', 'id1', 'id2', 'opening_id']) {
      const v = e.arguments?.[k]
      const n = typeof v === 'number' ? v : typeof v === 'string' && /^\d+$/.test(v) ? Number(v) : null
      if (n != null && !ids.includes(n)) ids.push(n)
    }
  }
  return ids
}

const SUGGESTION_KEYS = ['count', 'clearance', 'plumb', 'defects', 'freeSpace'] as const

// STAC answers ALOUD (USER 2026-11-14): on once the user has spoken to it, off when they mute it
const SPEAK_KEY = 'stac.voice.speak'
function readSpeakPref(): boolean | null {
  try { const v = localStorage.getItem(SPEAK_KEY); return v === '1' ? true : v === '0' ? false : null } catch { return null }
}
function writeSpeakPref(on: boolean) { try { localStorage.setItem(SPEAK_KEY, on ? '1' : '0') } catch { /* a private window */ } }
const VOICE_LANGS: readonly string[] = ['es', 'en', 'fr']

// how each user wants to be called — kept in this browser, per signed-in user
const nameKey = (user: string | null | undefined) => `stac.user.name.${user || 'anonymous'}`
function readName(user: string | null | undefined): string | null {
  try { return localStorage.getItem(nameKey(user)) || null } catch { return null }
}
function writeName(user: string | null | undefined, name: string | null) {
  try { if (name) localStorage.setItem(nameKey(user), name); else localStorage.removeItem(nameKey(user)) } catch { /* a private window */ }
}

function post(session: string | null, scan: string | null | undefined, path: string, body: Record<string, unknown>) {
  return fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: session, ...(scan ? { scan } : {}), ...body }) })
}

export default function AssistantPanel({ sessionId, scanKey, viewport, onVlmStatus, onFocusObjects, onVolumesChanged, onViewerAction, userKey, getUiState }: Props) {
  const api = useCallback((session: string | null, path: string, body: Record<string, unknown>) => post(session, scanKey, path, body), [scanKey])
  // the scene's objects reached the viewer for THIS session + scan (an answer never changes them)
  const objectsLoadedRef = useRef(false)
  const messagesRef = useRef<Message[]>([])
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
  // ── STAC's voice: the user speaks, the server writes it and STAC answers aloud ──
  const voice = useVoice(sessionId, scanKey)
  const [speakOn, setSpeakOn] = useState<boolean>(() => readSpeakPref() ?? false)
  const speakOnRef = useRef(speakOn)
  useEffect(() => { speakOnRef.current = speakOn }, [speakOn])
  const [voiceNote, setVoiceNote] = useState<string | null>(null)
  // ── the user's name: asked first, then used (USER 2026-11-14) ──
  const [userName, setUserName] = useState<string | null>(() => readName(userKey))
  const userNameRef = useRef(userName)
  useEffect(() => { userNameRef.current = userName }, [userName])
  useEffect(() => { const n = readName(userKey); setUserName(n); userNameRef.current = n }, [userKey])
  /** STAC has asked the name and the next thing said is the answer */
  const expectingNameRef = useRef(false)
  // the two commands this panel owns (the rest are the app's: App.tsx)
  useEffect(() => {
    const executors: Executors<'talk.set_user_name' | 'talk.forget_user_name'> = {
      'talk.set_user_name': ({ name }) => {
        const clean = String(name || '').trim()
        if (!clean) return { ok: false, reason: 'unavailable' }
        writeName(userKey, clean); setUserName(clean); userNameRef.current = clean; expectingNameRef.current = false
        return { ok: true }
      },
      'talk.forget_user_name': () => {
        writeName(userKey, null); setUserName(null); userNameRef.current = null; expectingNameRef.current = true
        return { ok: true }
      },
    }
    return registerCommands(executors)
  }, [userKey])
  // the voice engines' own state (they load in ~30 s after the server starts and leave the card
  // while a pipeline runs): the microphone waits for them, and a "loading" notice ends by itself
  const [voiceStatus, setVoiceStatus] = useState<'up' | 'loading' | 'busy' | 'down' | 'disabled' | null>(null)
  const voiceErrorKind = voice.error?.kind
  const clearVoiceError = voice.clearError
  useEffect(() => {
    if (voiceStatus === 'up' && (voiceErrorKind === 'loading' || voiceErrorKind === 'busy')) clearVoiceError()
  }, [voiceStatus, voiceErrorKind, clearVoiceError])

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
  }, [sessionId, viewport, api])

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    objectsLoadedRef.current = false
    api(sessionId, '/api/scene/objects', {}).then(r => r.json()).then(d => {
      if (!cancelled && d.objects) { viewport.current?.setAssistantObjects(d.objects); objectsLoadedRef.current = true }
    }).catch(() => { /* store may not exist yet */ })
    refreshVolumes()
    return () => { cancelled = true }
  }, [sessionId, viewport, refreshVolumes, api])

  // another scan (or session): the conversation was about the other one — it starts over
  useEffect(() => { setMessages(m => (expectingNameRef.current ? m : [])) }, [sessionId, scanKey])

  // the chat's measurements run on the GPU in their own process (semantic.chat.tools_device):
  // it is started when the chat opens on a scan, so the first measurement does not wait for it
  // (the server answers 'busy' while a pipeline has the card)
  useEffect(() => {
    if (!sessionId) return
    api(sessionId, '/api/spatial_qa/warmup', {}).catch(() => { /* the first question starts it */ })
  }, [sessionId, api])

  useEffect(() => { messagesRef.current = messages; scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' }) }, [messages])

  // CLOSING THE CHAT (its tab left or the inspector closed — the panel unmounts): the answers'
  // measurements leave the scene and the questions start over (USER 2026-10-09); the volumes
  // stay, they are the scene's
  useEffect(() => () => { viewport.current?.clearAssistantViz() }, [viewport])

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
        let voiceUp = true
        try {
          const vd = await (await fetch('/api/voice/status?start=true')).json()
          if (cancelled) return
          setVoiceStatus(vd.status)
          voiceUp = vd.status === 'up' || vd.status === 'disabled'
        } catch { /* an older server without the voice: the microphone stays off */ }
        setTimeout(tick, d.status === 'up' && voiceUp ? 30000 : 8000)
      } catch {
        if (!cancelled) setTimeout(tick, 15000)
      }
    }
    tick()
    return () => { cancelled = true }
     
  }, [])

  const ask = useCallback(async (question: string, spoken?: VoiceLang) => {
    if (!question.trim() || busy) return
    // the conversation so far goes with the question (the server keeps its last turns), so
    // "and the other one?" has a referent
    const history = messagesRef.current.filter(m => !m.pending && !m.error && m.text.trim()).map(m => ({ role: m.role, text: m.text }))
    setMessages(m => [...m, { role: 'user', text: question }, { role: 'assistant', text: '', pending: true }])
    setInput(''); setBusy(true)
    try {
      let d: Record<string, unknown>
      const deadline = Date.now() + 10 * 60_000
      for (;;) {
        // a spoken question is answered in the language it was spoken in
        const r = await api(sessionId, '/api/spatial_qa', {
          question, history, ...(spoken ? { lang: spoken } : {}),
          ...(userNameRef.current ? { user_name: userNameRef.current } : {}),
          ...(expectingNameRef.current ? { expecting: 'user_name' } : {}),
          ui: getUiState?.() ?? {},
        })
        d = await r.json()
        if (r.ok) break
        if (r.status !== 503 || d.status !== 'loading' || Date.now() > deadline) throw new Error(String(d.error || `HTTP ${r.status}`))
        setMessages(m => replaceLast(m, { role: 'assistant', text: '', pending: true, note: t('assistant.modelLoading') }))
        await new Promise(res => setTimeout(res, 8000))
      }
      const trace: TraceEntry[] = (d.tool_trace as TraceEntry[]) || []
      // the answer is shown FIRST; the objects are fetched only when the viewer does not hold
      // them yet (every answer re-read all of them before it appeared: seconds on a scan of
      // thousands of objects)
      const prov = d.provenance as { ok?: boolean; figures?: string[]; tools?: string[] } | undefined
      const unverified = prov && prov.ok === false ? [...(prov.figures || []), ...(prov.tools || [])] : undefined
      // THE COMMANDS OF THE INTERFACE the model chose (ui_actions): run here, in order, and
      // worded with the texts the server sent in the conversation's language
      const actions = (Array.isArray(d.ui_actions) ? d.ui_actions : []) as UiAction[]
      let answer = String(d.answer || '')
      if (actions.length) {
        const texts = (d.texts || { done_prefix: '', reasons: {} }) as AnswerTexts
        answer = (await runActions(actions, answer, texts, !!d.acted_only)).text
      }
      expectingNameRef.current = d.expecting === 'user_name'
      setMessages(m => replaceLast(m, { role: 'assistant', text: answer || t('assistant.noAnswer'), trace, unverified }))
      // … and said, in the answer's language (the server's: the question's)
      if (speakOnRef.current && answer) {
        const lang = VOICE_LANGS.includes(String(d.lang)) ? (d.lang as VoiceLang) : spoken
        if (lang) void voice.speak(answer, lang)
      }
      if (trace.length && !objectsLoadedRef.current) {
        try {
          const or = await api(sessionId, '/api/scene/objects', {})
          const od = await or.json()
          if (od.objects) { viewport.current?.setAssistantObjects(od.objects); objectsLoadedRef.current = true }
        } catch { /* keep whatever objects we had */ }
      }
      viewport.current?.visualizeMeasurement(trace)
      // the objects the tools were called on, then the ones the question named by id (the
      // server resolved them: an answer straight from that block calls no tool on them)
      const focusIds = objectsOfTrace(trace)
      for (const r of (Array.isArray(d.references) ? d.references as Array<{ id: number; exists: boolean }> : []))
        if (r.exists && !focusIds.includes(r.id)) focusIds.push(r.id)
      if (focusIds.length) onFocusObjects?.(focusIds)
      // the viewer actions, in the order the tools were called (after the focus: a travel or a
      // view the user asked for wins)
      for (const e of trace) {
        const act = (e.result as Record<string, unknown> | undefined)?.viewer
        if (act && typeof act === 'object') onViewerAction?.(act as Record<string, unknown>)
      }
      if (trace.some(x => x.tool === 'define_volume')) { await refreshVolumes(); onVolumesChanged?.() }
    } catch (e) {
      setMessages(m => replaceLast(m, { role: 'assistant', error: true, text: t('assistant.couldNotAnswer', { detail: e instanceof Error ? e.message : String(e) }) }))
    } finally {
      setBusy(false)
    }
  }, [sessionId, busy, viewport, refreshVolumes, t, onFocusObjects, onVolumesChanged, onViewerAction, api, voice, getUiState])

  // STAC ASKS THE NAME FIRST: once the model is up, to a user it does not know — written, and
  // said when the spoken answers are on. What the user says next is taken as the answer.
  const { lang: uiLang } = useI18n()
  const askedNameRef = useRef(false)
  useEffect(() => {
    if (vlmStatus !== 'up' || userName || askedNameRef.current || messagesRef.current.length) return
    askedNameRef.current = true
    expectingNameRef.current = true
    const text = t('assistant.askName')
    setMessages(m => (m.length ? m : [{ role: 'assistant', text }]))
    if (speakOnRef.current && VOICE_LANGS.includes(uiLang)) void voice.speak(text, uiLang as VoiceLang)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vlmStatus, userName])

  // THE MICROPHONE: one press starts listening, the next sends what was said (the recording
  // also ends by itself at the server's bound). What STAC heard is asked as a typed question is.
  const toggleMic = useCallback(async () => {
    if (voice.recording) { voice.stop(); return }
    setVoiceNote(null)
    const heard = await voice.start()
    if (!heard) return
    if (heard.noSpeech) { setVoiceNote(t('assistant.voice.noSpeech')); return }
    if (readSpeakPref() == null) { setSpeakOn(true); speakOnRef.current = true; writeSpeakPref(true) }
    ask(heard.text, heard.language)
  }, [voice, ask, t])
  const toggleSpeak = useCallback(() => {
    setSpeakOn(on => { writeSpeakPref(!on); if (on) voice.stopSpeaking(); return !on })
  }, [voice])
  const voiceError = voice.error ? t(`assistant.voice.error.${voice.error.kind}`) : null

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
      onVolumesChanged?.()
      setShowVolForm(false)
      ask(t('assistant.evaluateVolumeQuestion', { name: v.name }))
    }
  }, [sessionId, volForm, viewport, ask, t, onVolumesChanged, api])

  const removeVolume = useCallback(async (id: number) => {
    if (!sessionId) return
    await api(sessionId, '/api/scene/volumes/delete', { volume_id: id })
    setVolumes(prev => prev.filter(v => v.volume_id !== id))
    viewport.current?.removeUserVolume(id)
    onVolumesChanged?.()
  }, [sessionId, viewport, onVolumesChanged, api])

  const inputPlaceholder = voice.recording ? t('assistant.voice.listening') : voice.transcribing ? t('assistant.voice.transcribing') : vlmStatus === 'up' ? t('assistant.placeholder') : vlmStatus === 'loading' ? t('assistant.modelLoading') : vlmStatus === 'busy' ? t('assistant.modelBusy') : t('assistant.modelStarting')

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
                {m.unverified && m.unverified.length > 0 && (
                  <Banner tone="warn" compact>{t('assistant.unverified', { items: m.unverified.join(', ') })}</Banner>
                )}
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

      {(voiceError || voiceNote) && (
        <Banner tone="warn" compact>{voiceError ?? voiceNote}</Banner>
      )}
      <form className="stac-assistant__input" onSubmit={e => { e.preventDefault(); ask(input) }}>
        <IconButton type="button" variant={voice.recording ? 'danger' : 'ghost'} label={voice.recording ? t('assistant.voice.stop') : t('assistant.voice.start')}
          icon={voice.recording ? <Square aria-hidden /> : <Mic aria-hidden />} loading={voice.transcribing}
          disabled={!voiceSupported() || busy || voice.transcribing || vlmStatus !== 'up' || voiceStatus !== 'up'} onClick={toggleMic} />
        <Input value={input} disabled={busy || vlmStatus !== 'up' || voice.recording} placeholder={inputPlaceholder} aria-label={t('assistant.placeholder')} onChange={e => setInput(e.target.value)} />
        <IconButton type="button" variant="ghost" label={speakOn ? t('assistant.voice.speakOff') : t('assistant.voice.speakOn')}
          icon={speakOn ? <Volume2 aria-hidden /> : <VolumeX aria-hidden />} onClick={toggleSpeak} />
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
