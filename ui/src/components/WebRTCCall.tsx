/**
 * STAC Build — WebRTC Video/Audio Call Component
 * Peer-to-peer calls via existing WebSocket signaling
 * Hernán Barreto — Ingerop IN3
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { Mic, MicOff, Phone, PhoneOff, Video, VideoOff } from 'lucide-react'
import { Button } from './ui/Button'
import { IconButton } from './ui/IconButton'
import { useT } from '../i18n'

const ICE_CONFIG: RTCConfiguration = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
  ],
}

interface WebRTCCallProps {
  /** WebSocket ref used for signaling (the team WS) */
  wsRef: React.RefObject<WebSocket | null>
  /** Our user_id */
  userId: number
  /** Target user to call (null = no active call) */
  callTarget: { userId: number; username: string } | null
  /** Incoming call data */
  incomingCall: { from: number; username: string; callId: string; media: string } | null
  /** Close callback */
  onClose: () => void
  /** Clear incoming call */
  onIncomingHandled: () => void
}

export default function WebRTCCall({ wsRef, callTarget, incomingCall, onClose, onIncomingHandled }: WebRTCCallProps) {
  const t = useT()
  const [callState, setCallState] = useState<'idle' | 'ringing' | 'incoming' | 'connected' | 'ended'>('idle')
  const [isMuted, setIsMuted] = useState(false)
  const [isCameraOff, setIsCameraOff] = useState(false)
  const [callId, setCallId] = useState<string | null>(null)
  const [remoteName, setRemoteName] = useState('')

  const pcRef = useRef<RTCPeerConnection | null>(null)
  const localStreamRef = useRef<MediaStream | null>(null)
  const localVideoRef = useRef<HTMLVideoElement>(null)
  const remoteVideoRef = useRef<HTMLVideoElement>(null)
  const pendingCandidates = useRef<RTCIceCandidateInit[]>([])

  const cleanup = useCallback(() => {
    localStreamRef.current?.getTracks().forEach(tr => tr.stop())
    localStreamRef.current = null
    pcRef.current?.close()
    pcRef.current = null
    setCallState('ended')
    setTimeout(() => { onClose() }, 1500)
  }, [onClose])

  const sendSignal = useCallback((msg: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify(msg))
  }, [wsRef])

  const createPeerConnection = useCallback(() => {
    const pc = new RTCPeerConnection(ICE_CONFIG)
    pc.onicecandidate = (e) => {
      if (e.candidate && callId) sendSignal({ type: 'rtc_ice', to: callTarget?.userId ?? incomingCall?.from, call_id: callId, candidate: e.candidate.toJSON() })
    }
    pc.ontrack = (e) => { if (remoteVideoRef.current && e.streams[0]) remoteVideoRef.current.srcObject = e.streams[0] }
    pc.onconnectionstatechange = () => { if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') cleanup() }
    pcRef.current = pc
    return pc
  }, [callId, callTarget, incomingCall, sendSignal, cleanup])

  const getLocalMedia = useCallback(async (video: boolean = true) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: video ? { width: 640, height: 480 } : false })
      localStreamRef.current = stream
      if (localVideoRef.current) localVideoRef.current.srcObject = stream
      return stream
    } catch (e) {
      console.error('[WebRTC] Failed to get media:', e)
      throw e
    }
  }, [])

  // Initiate outgoing call
  useEffect(() => {
    if (!callTarget || callState !== 'idle') return
    const cid = `call-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    setCallId(cid)
    setRemoteName(callTarget.username)
    setCallState('ringing')
    sendSignal({ type: 'call_invite', to: callTarget.userId, call_id: cid, media: 'video' })
  }, [callTarget, callState, sendSignal])

  // Handle incoming call
  useEffect(() => {
    if (incomingCall && callState === 'idle') {
      setCallState('incoming')
      setCallId(incomingCall.callId)
      setRemoteName(incomingCall.username)
    }
  }, [incomingCall, callState])

  const acceptCall = useCallback(async () => {
    if (!incomingCall || !callId) return
    onIncomingHandled()
    sendSignal({ type: 'call_accept', to: incomingCall.from, call_id: callId })
    const stream = await getLocalMedia(incomingCall.media === 'video')
    const pc = createPeerConnection()
    stream.getTracks().forEach(tr => pc.addTrack(tr, stream))
    setCallState('connected')
    for (const c of pendingCandidates.current) await pc.addIceCandidate(new RTCIceCandidate(c))
    pendingCandidates.current = []
  }, [incomingCall, callId, getLocalMedia, createPeerConnection, sendSignal, onIncomingHandled])

  const declineCall = useCallback(() => {
    if (incomingCall && callId) sendSignal({ type: 'call_decline', to: incomingCall.from, call_id: callId })
    onIncomingHandled()
    onClose()
  }, [incomingCall, callId, sendSignal, onIncomingHandled, onClose])

  const endCall = useCallback(() => {
    const target = callTarget?.userId ?? incomingCall?.from
    if (callId && target) sendSignal({ type: 'call_end', to: target, call_id: callId })
    cleanup()
  }, [callId, callTarget, incomingCall, sendSignal, cleanup])

  // Listen for signaling messages
  useEffect(() => {
    const ws = wsRef.current
    if (!ws) return
    const handler = (event: MessageEvent) => {
      let msg: Record<string, unknown>
      try { msg = JSON.parse(event.data) } catch { return }
      if (msg.call_id && msg.call_id !== callId && callState !== 'idle') return
      switch (msg.type) {
        case 'call_accept': {
          void (async () => {
            const stream = await getLocalMedia(true)
            const pc = createPeerConnection()
            stream.getTracks().forEach(tr => pc.addTrack(tr, stream))
            const offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            sendSignal({ type: 'rtc_offer', to: callTarget?.userId, call_id: callId, sdp: offer.sdp })
            setCallState('connected')
          })()
          break
        }
        case 'call_decline':
        case 'call_end':
          cleanup()
          break
        case 'rtc_offer': {
          const pc = pcRef.current
          if (!pc) break
          void (async () => {
            await pc.setRemoteDescription(new RTCSessionDescription({ type: 'offer', sdp: msg.sdp as string }))
            const answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            sendSignal({ type: 'rtc_answer', to: msg.from as number, call_id: callId, sdp: answer.sdp })
            for (const c of pendingCandidates.current) await pc.addIceCandidate(new RTCIceCandidate(c))
            pendingCandidates.current = []
          })()
          break
        }
        case 'rtc_answer': {
          const pc = pcRef.current
          if (!pc) break
          pc.setRemoteDescription(new RTCSessionDescription({ type: 'answer', sdp: msg.sdp as string }))
          break
        }
        case 'rtc_ice': {
          const candidate = msg.candidate as RTCIceCandidateInit
          const pc = pcRef.current
          if (pc?.remoteDescription) pc.addIceCandidate(new RTCIceCandidate(candidate))
          else pendingCandidates.current.push(candidate)
          break
        }
      }
    }
    ws.addEventListener('message', handler)
    return () => ws.removeEventListener('message', handler)
  }, [wsRef, callId, callState, callTarget, getLocalMedia, createPeerConnection, sendSignal, cleanup])

  const toggleMute = () => {
    const stream = localStreamRef.current
    if (!stream) return
    stream.getAudioTracks().forEach(tr => { tr.enabled = !tr.enabled })
    setIsMuted(prev => !prev)
  }

  const toggleCamera = () => {
    const stream = localStreamRef.current
    if (!stream) return
    stream.getVideoTracks().forEach(tr => { tr.enabled = !tr.enabled })
    setIsCameraOff(prev => !prev)
  }

  if (callState === 'incoming') {
    return (
      <div className="stac-call__incoming" role="alertdialog" aria-label={t('call.incoming')}>
        <Video className="stac-call__incoming-icon" aria-hidden />
        <div className="stac-call__name">{remoteName}</div>
        <div className="stac-call__label">{t('call.incoming')}</div>
        <div className="stac-call__actions">
          <Button variant="primary" icon={<Phone aria-hidden />} onClick={acceptCall}>{t('call.accept')}</Button>
          <Button variant="danger" icon={<PhoneOff aria-hidden />} onClick={declineCall}>{t('call.decline')}</Button>
        </div>
      </div>
    )
  }

  if (callState === 'ringing') {
    return (
      <div className="stac-call__overlay" role="dialog" aria-label={t('call.calling', { name: remoteName })}>
        <div className="stac-call__ringing">
          <span className="stac-call__pulse" aria-hidden />
          <div className="stac-call__name">{t('call.calling', { name: remoteName })}</div>
          <Button variant="danger" icon={<PhoneOff aria-hidden />} onClick={endCall}>{t('common.cancel')}</Button>
        </div>
      </div>
    )
  }

  if (callState === 'connected' || callState === 'ended') {
    return (
      <div className="stac-call__overlay" role="dialog" aria-label={t('call.inCall', { name: remoteName })}>
        <div className="stac-call__stage">
          <video ref={remoteVideoRef} className="stac-call__remote" autoPlay playsInline />
          <video ref={localVideoRef} className="stac-call__local" autoPlay playsInline muted />
          {callState === 'ended' && <div className="stac-call__ended">{t('call.ended')}</div>}
          <div className="stac-call__controls">
            <IconButton size="lg" variant="secondary" active={isMuted} label={isMuted ? t('call.unmute') : t('call.mute')} icon={isMuted ? <MicOff aria-hidden /> : <Mic aria-hidden />} onClick={toggleMute} />
            <IconButton size="lg" variant="secondary" active={isCameraOff} label={isCameraOff ? t('call.cameraOn') : t('call.cameraOff')} icon={isCameraOff ? <VideoOff aria-hidden /> : <Video aria-hidden />} onClick={toggleCamera} />
            <IconButton size="lg" variant="danger" label={t('call.end')} icon={<PhoneOff aria-hidden />} onClick={endCall} />
          </div>
        </div>
      </div>
    )
  }

  return null
}
