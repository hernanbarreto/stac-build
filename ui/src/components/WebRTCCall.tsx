/**
 * STAC Build — WebRTC Video/Audio Call Component
 * Peer-to-peer calls via existing WebSocket signaling
 * Hernán Barreto — Ingerop IN3
 */
import { useState, useEffect, useRef, useCallback } from 'react'

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

export default function WebRTCCall({
    wsRef, userId: _userId, callTarget, incomingCall, onClose, onIncomingHandled,
}: WebRTCCallProps) {
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

    // ── Cleanup ─────────────────────────────────────────────────
    const cleanup = useCallback(() => {
        localStreamRef.current?.getTracks().forEach(t => t.stop())
        localStreamRef.current = null
        pcRef.current?.close()
        pcRef.current = null
        setCallState('ended')
        setTimeout(() => {
            onClose()
        }, 1500)
    }, [onClose])

    // ── Send signaling ──────────────────────────────────────────
    const sendSignal = useCallback((msg: Record<string, unknown>) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(msg))
        }
    }, [wsRef])

    // ── Create peer connection ──────────────────────────────────
    const createPeerConnection = useCallback(() => {
        const pc = new RTCPeerConnection(ICE_CONFIG)

        pc.onicecandidate = (e) => {
            if (e.candidate && callId) {
                sendSignal({
                    type: 'rtc_ice',
                    to: callTarget?.userId ?? incomingCall?.from,
                    call_id: callId,
                    candidate: e.candidate.toJSON(),
                })
            }
        }

        pc.ontrack = (e) => {
            if (remoteVideoRef.current && e.streams[0]) {
                remoteVideoRef.current.srcObject = e.streams[0]
            }
        }

        pc.onconnectionstatechange = () => {
            if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
                cleanup()
            }
        }

        pcRef.current = pc
        return pc
    }, [callId, callTarget, incomingCall, sendSignal, cleanup])

    // ── Get local media ─────────────────────────────────────────
    const getLocalMedia = useCallback(async (video: boolean = true) => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: true,
                video: video ? { width: 640, height: 480 } : false,
            })
            localStreamRef.current = stream
            if (localVideoRef.current) {
                localVideoRef.current.srcObject = stream
            }
            return stream
        } catch (e) {
            console.error('[WebRTC] Failed to get media:', e)
            throw e
        }
    }, [])

    // ── Initiate outgoing call ──────────────────────────────────
    useEffect(() => {
        if (!callTarget || callState !== 'idle') return

        const cid = `call-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
        setCallId(cid)
        setRemoteName(callTarget.username)
        setCallState('ringing')

        sendSignal({
            type: 'call_invite',
            to: callTarget.userId,
            call_id: cid,
            media: 'video',
        })
    }, [callTarget, callState, sendSignal])

    // ── Handle incoming call ────────────────────────────────────
    useEffect(() => {
        if (incomingCall && callState === 'idle') {
            setCallState('incoming')
            setCallId(incomingCall.callId)
            setRemoteName(incomingCall.username)
        }
    }, [incomingCall, callState])

    // ── Accept incoming call ────────────────────────────────────
    const acceptCall = useCallback(async () => {
        if (!incomingCall || !callId) return
        onIncomingHandled()

        sendSignal({ type: 'call_accept', to: incomingCall.from, call_id: callId })

        const stream = await getLocalMedia(incomingCall.media === 'video')
        const pc = createPeerConnection()
        stream.getTracks().forEach(t => pc.addTrack(t, stream))

        setCallState('connected')

        // Process any pending ICE candidates
        for (const c of pendingCandidates.current) {
            await pc.addIceCandidate(new RTCIceCandidate(c))
        }
        pendingCandidates.current = []
    }, [incomingCall, callId, getLocalMedia, createPeerConnection, sendSignal, onIncomingHandled])

    // ── Decline incoming call ───────────────────────────────────
    const declineCall = useCallback(() => {
        if (incomingCall && callId) {
            sendSignal({ type: 'call_decline', to: incomingCall.from, call_id: callId })
        }
        onIncomingHandled()
        onClose()
    }, [incomingCall, callId, sendSignal, onIncomingHandled, onClose])

    // ── End call ────────────────────────────────────────────────
    const endCall = useCallback(() => {
        const target = callTarget?.userId ?? incomingCall?.from
        if (callId && target) {
            sendSignal({ type: 'call_end', to: target, call_id: callId })
        }
        cleanup()
    }, [callId, callTarget, incomingCall, sendSignal, cleanup])

    // ── Listen for signaling messages ───────────────────────────
    useEffect(() => {
        const ws = wsRef.current
        if (!ws) return

        const handler = (event: MessageEvent) => {
            let msg: Record<string, unknown>
            try { msg = JSON.parse(event.data) } catch { return }

            if (msg.call_id && msg.call_id !== callId && callState !== 'idle') return

            switch (msg.type) {
                case 'call_accept': {
                    // Our outgoing call was accepted — create offer
                    ; (async () => {
                        const stream = await getLocalMedia(true)
                        const pc = createPeerConnection()
                        stream.getTracks().forEach(t => pc.addTrack(t, stream))

                        const offer = await pc.createOffer()
                        await pc.setLocalDescription(offer)
                        sendSignal({
                            type: 'rtc_offer',
                            to: callTarget?.userId,
                            call_id: callId,
                            sdp: offer.sdp,
                        })
                        setCallState('connected')
                    })()
                    break
                }

                case 'call_decline':
                case 'call_end':
                    cleanup()
                    break

                case 'rtc_offer': {
                    // Incoming offer — create answer
                    const pc = pcRef.current
                    if (!pc) break
                        ; (async () => {
                            await pc.setRemoteDescription(new RTCSessionDescription({
                                type: 'offer',
                                sdp: msg.sdp as string,
                            }))
                            const answer = await pc.createAnswer()
                            await pc.setLocalDescription(answer)
                            sendSignal({
                                type: 'rtc_answer',
                                to: msg.from as number,
                                call_id: callId,
                                sdp: answer.sdp,
                            })
                            // Process pending
                            for (const c of pendingCandidates.current) {
                                await pc.addIceCandidate(new RTCIceCandidate(c))
                            }
                            pendingCandidates.current = []
                        })()
                    break
                }

                case 'rtc_answer': {
                    const pc = pcRef.current
                    if (!pc) break
                    pc.setRemoteDescription(new RTCSessionDescription({
                        type: 'answer',
                        sdp: msg.sdp as string,
                    }))
                    break
                }

                case 'rtc_ice': {
                    const candidate = msg.candidate as RTCIceCandidateInit
                    const pc = pcRef.current
                    if (pc?.remoteDescription) {
                        pc.addIceCandidate(new RTCIceCandidate(candidate))
                    } else {
                        pendingCandidates.current.push(candidate)
                    }
                    break
                }
            }
        }

        ws.addEventListener('message', handler)
        return () => ws.removeEventListener('message', handler)
    }, [wsRef, callId, callState, callTarget, getLocalMedia, createPeerConnection, sendSignal, cleanup])

    // ── Toggle controls ─────────────────────────────────────────
    const toggleMute = () => {
        const stream = localStreamRef.current
        if (!stream) return
        stream.getAudioTracks().forEach(t => { t.enabled = !t.enabled })
        setIsMuted(prev => !prev)
    }

    const toggleCamera = () => {
        const stream = localStreamRef.current
        if (!stream) return
        stream.getVideoTracks().forEach(t => { t.enabled = !t.enabled })
        setIsCameraOff(prev => !prev)
    }

    // ── Incoming call ring UI ───────────────────────────────────
    if (callState === 'incoming') {
        return (
            <div className="webrtc-incoming">
                <div className="webrtc-incoming-card">
                    <div className="webrtc-incoming-avatar">📹</div>
                    <div className="webrtc-incoming-name">{remoteName}</div>
                    <div className="webrtc-incoming-label">Incoming call...</div>
                    <div className="webrtc-incoming-actions">
                        <button className="webrtc-btn-accept" onClick={acceptCall}>✅ Accept</button>
                        <button className="webrtc-btn-decline" onClick={declineCall}>❌ Decline</button>
                    </div>
                </div>
            </div>
        )
    }

    // ── Ringing UI ──────────────────────────────────────────────
    if (callState === 'ringing') {
        return (
            <div className="webrtc-overlay">
                <div className="webrtc-ringing">
                    <div className="webrtc-ringing-pulse" />
                    <div className="webrtc-ringing-name">Calling {remoteName}...</div>
                    <button className="webrtc-btn-end" onClick={endCall}>Cancel</button>
                </div>
            </div>
        )
    }

    // ── Connected call UI ───────────────────────────────────────
    if (callState === 'connected' || callState === 'ended') {
        return (
            <div className="webrtc-overlay">
                <div className="webrtc-call-container">
                    <video
                        ref={remoteVideoRef}
                        className="webrtc-remote-video"
                        autoPlay
                        playsInline
                    />
                    <video
                        ref={localVideoRef}
                        className="webrtc-local-video"
                        autoPlay
                        playsInline
                        muted
                    />
                    {callState === 'ended' && (
                        <div className="webrtc-ended-overlay">Call ended</div>
                    )}
                    <div className="webrtc-controls">
                        <button
                            className={`webrtc-ctrl-btn ${isMuted ? 'active' : ''}`}
                            onClick={toggleMute}
                            title={isMuted ? 'Unmute' : 'Mute'}
                        >
                            {isMuted ? '🔇' : '🎤'}
                        </button>
                        <button
                            className={`webrtc-ctrl-btn ${isCameraOff ? 'active' : ''}`}
                            onClick={toggleCamera}
                            title={isCameraOff ? 'Camera On' : 'Camera Off'}
                        >
                            {isCameraOff ? '🚫' : '📹'}
                        </button>
                        <button className="webrtc-ctrl-btn end" onClick={endCall} title="End Call">
                            📞
                        </button>
                    </div>
                </div>
            </div>
        )
    }

    return null
}
