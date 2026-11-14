// STAC-Builder — HANDS-FREE listening: the microphone stays open and every utterance is cut out
// and handed over; the server writes it and says whether it names STAC ("STAC, …" — voice/wake.py).
// What does not name STAC is dropped: nothing of it reaches the model.
//
// USER 2026-11-14: "el usuario debe poder hablar con stac build … tanto para entrar comandos como
// para conversar con nosotros" / "el sistema se llama stac".
//
// The browser only decides WHERE AN UTTERANCE STARTS AND ENDS (the level of the microphone
// against the room's own noise); whether it held speech at all is the server's voice-activity
// detector's verdict. The numbers below are this endpointing's, adjustable — a wrong cut costs one
// request or a clipped sentence, never a wrong answer.
import { useEffect, useRef, useState } from 'react'

const TARGET_RATE = 16000        // what the recogniser reads
const START_MS = 90              // speech this long starts an utterance
const END_SILENCE_MS = 800       // silence this long ends it (a pause to breathe is shorter)
const PRE_ROLL_MS = 320          // kept from before the start: the first syllable is not lost
const MIN_SPEECH_MS = 250        // shorter than this was a click or a cough
const MAX_UTTERANCE_MS = 15000   // one command or question (the server's bound is 30 s)
const NOISE_FACTOR = 3           // speech = this many times the room's own level …
const ABS_FLOOR = 0.008          // … and never under this (about -42 dBFS)

function encodeWav(samples: Float32Array, rate: number): Blob {
  const buf = new ArrayBuffer(44 + samples.length * 2)
  const v = new DataView(buf)
  const text = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)) }
  text(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); text(8, 'WAVE'); text(12, 'fmt ')
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true)
  v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); text(36, 'data')
  v.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true)
  return new Blob([buf], { type: 'audio/wav' })
}

/** Linear resampling to the recogniser's rate (only when the browser could not capture at it). */
function resample(x: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return x
  const n = Math.floor(x.length * to / from)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const p = i * from / to, k = Math.floor(p), f = p - k
    out[i] = x[k] * (1 - f) + (x[Math.min(k + 1, x.length - 1)]) * f
  }
  return out
}

export type HandsFreeError = 'unsupported' | 'denied' | 'no-microphone' | 'failed'

/**
 * While `enabled`, listen and call `onUtterance` with each utterance as a 16 kHz mono WAV.
 * While `paused` (STAC is speaking, or a turn is being answered) the microphone is ignored.
 */
export function useHandsFree(enabled: boolean, paused: boolean, onUtterance: (wav: Blob) => void) {
  const [listening, setListening] = useState(false)
  const [hearing, setHearing] = useState(false)          // an utterance is being captured now
  const [error, setError] = useState<HandsFreeError | null>(null)
  const pausedRef = useRef(paused)
  const cbRef = useRef(onUtterance)
  useEffect(() => { pausedRef.current = paused }, [paused])
  useEffect(() => { cbRef.current = onUtterance }, [onUtterance])

  useEffect(() => {
    if (!enabled) return
    let stopped = false
    let stream: MediaStream | null = null
    let ctx: AudioContext | null = null
    let node: ScriptProcessorNode | null = null
    setError(null)
    ;(async () => {
      if (!navigator.mediaDevices?.getUserMedia || typeof AudioContext === 'undefined') { setError('unsupported'); return }
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } })
      } catch (e) {
        const name = (e as DOMException)?.name
        setError(name === 'NotAllowedError' || name === 'SecurityError' ? 'denied' : name === 'NotFoundError' ? 'no-microphone' : 'failed')
        return
      }
      if (stopped) { stream.getTracks().forEach(t => t.stop()); return }
      try { ctx = new AudioContext({ sampleRate: TARGET_RATE }) } catch { ctx = new AudioContext() }
      const rate = ctx.sampleRate
      const size = rate <= 16000 ? 512 : rate <= 32000 ? 1024 : 2048       // ~32-43 ms a frame
      const frameMs = size / rate * 1000
      const source = ctx.createMediaStreamSource(stream)
      node = ctx.createScriptProcessor(size, 1, 1)
      const preRoll: Float32Array[] = []
      let chunks: Float32Array[] = []
      let noise = ABS_FLOOR, inSpeech = false, speechRun = 0, silenceMs = 0, spokenMs = 0, lengthMs = 0
      const reset = () => { chunks = []; inSpeech = false; speechRun = 0; silenceMs = 0; spokenMs = 0; lengthMs = 0; setHearing(false) }
      node.onaudioprocess = ev => {
        const x = ev.inputBuffer.getChannelData(0)
        if (pausedRef.current) { if (inSpeech) reset(); preRoll.length = 0; return }
        let sum = 0
        for (let i = 0; i < x.length; i++) sum += x[i] * x[i]
        const rms = Math.sqrt(sum / x.length)
        const speech = rms > Math.max(noise * NOISE_FACTOR, ABS_FLOOR)
        const frame = new Float32Array(x)
        if (!inSpeech) {
          if (!speech) noise = 0.95 * noise + 0.05 * rms                 // the room, when nobody speaks
          preRoll.push(frame)
          while (preRoll.length * frameMs > PRE_ROLL_MS + START_MS) preRoll.shift()
          speechRun = speech ? speechRun + 1 : 0
          if (speechRun * frameMs >= START_MS) {
            inSpeech = true; chunks = preRoll.splice(0); lengthMs = chunks.length * frameMs
            spokenMs = speechRun * frameMs; silenceMs = 0; setHearing(true)
          }
          return
        }
        chunks.push(frame); lengthMs += frameMs
        if (speech) { spokenMs += frameMs; silenceMs = 0 } else silenceMs += frameMs
        if (silenceMs >= END_SILENCE_MS || lengthMs >= MAX_UTTERANCE_MS) {
          const enough = spokenMs >= MIN_SPEECH_MS
          const total = chunks.reduce((n, c) => n + c.length, 0)
          const all = new Float32Array(total)
          let o = 0
          for (const c of chunks) { all.set(c, o); o += c.length }
          reset()
          if (enough) cbRef.current(encodeWav(resample(all, rate, TARGET_RATE), TARGET_RATE))
        }
      }
      source.connect(node)
      node.connect(ctx.destination)          // a script processor only runs when connected; it outputs silence
      setListening(true)
    })()
    return () => {
      stopped = true
      if (node) { node.onaudioprocess = null; try { node.disconnect() } catch { /* already */ } }
      stream?.getTracks().forEach(t => t.stop())
      void ctx?.close().catch(() => { /* already closed */ })
      setListening(false); setHearing(false)
    }
  }, [enabled])

  return { listening, hearing, error }
}
