// STAC-Builder — STAC's voice in the browser: record what the user says, have the server write
// it (POST /api/voice/transcribe: Whisper on the card, the language detected among es / en / fr)
// and play what STAC answers (POST /api/voice/speak: Kokoro on the card).
//
// USER 2026-11-14: "incorpora herramientas para hablarle y que traduzca a texto y que responda
// con voz sintetica en los 3 idiomas del sistema".
//
// Nothing is recognised or synthesised in the browser and no audio leaves the pod: the browser
// only records (MediaRecorder) and plays.
import { useCallback, useEffect, useRef, useState } from 'react'

export type VoiceLang = 'es' | 'en' | 'fr'

export interface Heard {
  text: string
  language: VoiceLang
  /** the recogniser's detector found no speech in the recording */
  noSpeech: boolean
  audioSeconds: number
  /** what was heard names STAC ("STAC, …" — the server's voice/wake.py) */
  addressed: boolean
  /** what was said TO STAC: the text without its name ("" when only the name was said) */
  command: string
  /** what STAC says when only its name was said ("Dime."), in the language heard */
  ack: string
}

export type VoiceError = 'unsupported' | 'denied' | 'no-microphone' | 'busy' | 'loading' | 'failed'

/** the longest recording sent (the server's voice.stt.max_audio_s is 30 s: one command or question) */
const MAX_RECORDING_MS = 29_000

function pickMime(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined
  for (const m of ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm', 'audio/mp4'])
    if (MediaRecorder.isTypeSupported(m)) return m
  return ''
}

export function voiceSupported(): boolean {
  return typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== 'undefined'
}

export function useVoice(sessionId: string | null, scanKey?: string | null, getNames?: () => string[]) {
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const [error, setError] = useState<{ kind: VoiceError; detail?: string } | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)
  const resolveRef = useRef<((h: Heard | null) => void) | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const urlRef = useRef<string | null>(null)

  const releaseMic = useCallback(() => {
    if (timerRef.current != null) { window.clearTimeout(timerRef.current); timerRef.current = null }
    streamRef.current?.getTracks().forEach(tr => tr.stop())
    streamRef.current = null
    recorderRef.current = null
  }, [])

  const stopSpeaking = useCallback(() => {
    const a = audioRef.current
    if (a) { a.pause(); a.src = '' }
    audioRef.current = null
    if (urlRef.current) { URL.revokeObjectURL(urlRef.current); urlRef.current = null }
    setSpeaking(false)
  }, [])

  const transcribe = useCallback(async (blob: Blob): Promise<Heard | null> => {
    setTranscribing(true)
    try {
      const q = new URLSearchParams()
      if (sessionId) q.set('session_id', sessionId)
      if (scanKey) q.set('scan', scanKey)
      // the names the recogniser should know how to write: the sessions on the server, the scans
      const names = getNames?.() ?? []
      if (names.length) q.set('vocab', names.slice(0, 40).join(','))
      const r = await fetch(`/api/voice/transcribe?${q.toString()}`, { method: 'POST', headers: { 'Content-Type': blob.type || 'application/octet-stream' }, body: blob })
      const d = await r.json().catch(() => ({}))
      if (!r.ok) {
        const kind: VoiceError = d.status === 'busy' ? 'busy' : d.status === 'loading' ? 'loading' : 'failed'
        setError({ kind, detail: String(d.error || `HTTP ${r.status}`) })
        return null
      }
      return {
        text: String(d.text || ''), language: (d.language as VoiceLang) || 'es', noSpeech: !!d.no_speech || !String(d.text || '').trim(),
        audioSeconds: Number(d.audio_s || 0), addressed: !!d.addressed, command: String(d.command ?? d.text ?? ''), ack: String(d.ack || ''),
      }
    } catch (e) {
      setError({ kind: 'failed', detail: e instanceof Error ? e.message : String(e) })
      return null
    } finally {
      setTranscribing(false)
    }
  }, [sessionId, scanKey, getNames])

  /** Start recording. Resolves — when stop() is called or the bound is reached — with what was heard. */
  const start = useCallback(async (): Promise<Heard | null> => {
    setError(null)
    if (!voiceSupported()) { setError({ kind: 'unsupported' }); return null }
    if (recorderRef.current) return null
    stopSpeaking()                                    // STAC stops talking when the user speaks
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } })
    } catch (e) {
      const name = (e as DOMException)?.name
      setError({ kind: name === 'NotAllowedError' || name === 'SecurityError' ? 'denied' : name === 'NotFoundError' ? 'no-microphone' : 'failed', detail: e instanceof Error ? e.message : String(e) })
      return null
    }
    streamRef.current = stream
    const mime = pickMime()
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
    recorderRef.current = rec
    chunksRef.current = []
    rec.ondataavailable = ev => { if (ev.data.size) chunksRef.current.push(ev.data) }
    const done = new Promise<Heard | null>(resolve => { resolveRef.current = resolve })
    rec.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: rec.mimeType || mime || 'audio/webm' })
      releaseMic()
      setRecording(false)
      const resolve = resolveRef.current
      resolveRef.current = null
      resolve?.(blob.size ? await transcribe(blob) : null)
    }
    rec.start()
    setRecording(true)
    timerRef.current = window.setTimeout(() => { if (recorderRef.current?.state === 'recording') recorderRef.current.stop() }, MAX_RECORDING_MS)
    return done
  }, [releaseMic, stopSpeaking, transcribe])

  /** End the recording: start()'s promise then resolves with the transcription. */
  const stop = useCallback(() => {
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
  }, [])

  /** Throw the recording away (nothing is sent). */
  const cancel = useCallback(() => {
    const rec = recorderRef.current
    if (!rec) return
    rec.onstop = null
    try { if (rec.state === 'recording') rec.stop() } catch { /* already stopped */ }
    releaseMic()
    setRecording(false)
    resolveRef.current?.(null)
    resolveRef.current = null
  }, [releaseMic])

  /** STAC says ``text`` in ``lang``. Resolves when the audio ends (or could not be played). */
  const speak = useCallback(async (text: string, lang: VoiceLang): Promise<void> => {
    if (!text.trim()) return
    stopSpeaking()
    try {
      const r = await fetch('/api/voice/speak', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, lang }) })
      if (!r.ok) {
        const d = await r.json().catch(() => ({}))
        setError({ kind: d.status === 'busy' ? 'busy' : d.status === 'loading' ? 'loading' : 'failed', detail: String(d.error || `HTTP ${r.status}`) })
        return
      }
      const url = URL.createObjectURL(await r.blob())
      urlRef.current = url
      const a = new Audio(url)
      audioRef.current = a
      setSpeaking(true)
      await new Promise<void>(resolve => {
        a.onended = () => resolve()
        a.onerror = () => resolve()
        a.play().catch(() => resolve())              // autoplay refused: the text is on screen
      })
    } catch (e) {
      setError({ kind: 'failed', detail: e instanceof Error ? e.message : String(e) })
    } finally {
      if (audioRef.current) stopSpeaking()
    }
  }, [stopSpeaking])

  useEffect(() => () => { cancel(); stopSpeaking() }, [cancel, stopSpeaking])

  const clearError = useCallback(() => setError(null), [])

  return { recording, transcribing, speaking, error, clearError, start, stop, cancel, speak, stopSpeaking, transcribe }
}
