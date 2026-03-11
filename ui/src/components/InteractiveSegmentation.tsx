import { useState, useRef, useEffect, useCallback } from 'react'
import { useConfirmDialog } from './ConfirmDialog'

interface InstanceInfo {
    id: number
    label: string
    color: string
    total_points: number
    excluded?: boolean
}

interface Props {
    sessionId: string
    onClose: () => void
    onSuccess?: (instances: any[]) => void
    onUpdate?: () => void
}

type SegMode = 'manual' | 'text' | 'auto'

// Predefined colors for new instances
const INSTANCE_COLORS = [
    '#00d4aa', '#e94560', '#3498db', '#f39c12', '#9b59b6',
    '#2ecc71', '#e74c3c', '#1abc9c', '#e67e22', '#8e44ad',
]

export default function SegmentationManager({ sessionId, onClose, onUpdate }: Props) {
    const { confirmDanger, dialogElement } = useConfirmDialog()
    // State
    const [loading, setLoading] = useState(true)
    const [stateId, setStateId] = useState<string | null>(null)
    const [keyframes, setKeyframes] = useState<string[]>([])
    const [kfIndex, setKfIndex] = useState(0)
    const [status, setStatus] = useState('Initializing...')
    const [mode, setMode] = useState<SegMode>('manual')
    const [instances, setInstances] = useState<InstanceInfo[]>([])
    const [selectedInstance, setSelectedInstance] = useState<number | null>(null)
    const [maskOverlay, setMaskOverlay] = useState<string | null>(null)
    const [hasPrompts, setHasPrompts] = useState(false)
    const [labelName, setLabelName] = useState('')
    const [textPrompt, setTextPrompt] = useState('')
    const [promptPoints, setPromptPoints] = useState<{ x: number, y: number, label: number }[]>([])

    // Zoom/Pan state
    const [zoom, setZoom] = useState(1)
    const [pan, setPan] = useState({ x: 0, y: 0 })
    const [isPanning, setIsPanning] = useState(false)
    const panStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 })
    const mouseDownPos = useRef<{ x: number; y: number } | null>(null)
    const wasDragging = useRef(false)

    const imgRef = useRef<HTMLImageElement>(null)
    const canvasRef = useRef<HTMLDivElement>(null)
    const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
    const maskImageRef = useRef<HTMLImageElement | null>(null)
    const initRef = useRef(false)

    const currentFrame = keyframes[kfIndex] || null
    // SAM3 now loads only keyframes sequentially (via temp dir).
    // The SAM3 frame_idx is the position in the sorted keyframes list = kfIndex.
    const currentFrameIdx = kfIndex

    // ── Init ──────────────────────────────────────────────────
    useEffect(() => {
        if (initRef.current) return
        initRef.current = true

        const init = async () => {
            try {
                setLoading(true)

                // Load existing instances
                setStatus('Loading existing segmentation...')
                const segRes = await fetch(`/api/sessions/${sessionId}/segmentation`)
                if (segRes.ok) {
                    const segData = await segRes.json()
                    setInstances((segData.instances || []).map((inst: any, i: number) => ({
                        id: inst.id,
                        label: inst.label || `Object ${inst.id}`,
                        color: inst.color || INSTANCE_COLORS[i % INSTANCE_COLORS.length],
                        total_points: inst.total_points || 0,
                        excluded: inst.excluded || false,
                    })))
                }

                // Load keyframes
                setStatus('Loading keyframes...')
                const kfRes = await fetch(`/api/sessions/${sessionId}/keyframes`)
                if (!kfRes.ok) throw new Error('Failed to load keyframes')
                const kfData = await kfRes.json()
                const kfList: string[] = kfData.keyframes || []
                if (kfList.length === 0) throw new Error('No keyframes found')
                setKeyframes(kfList)
                setKfIndex(0)

                // Start SAM3 session
                setStatus(`Initializing segmentation (${kfList.length} keyframes)...`)
                const res = await fetch(`/api/segmentation/start_session/${sessionId}`, { method: 'POST' })
                if (!res.ok) throw new Error('Failed to init segmentation')
                const data = await res.json()
                setStateId(data.state_id)

                setStatus(`Ready — ${kfList.length} keyframes`)
            } catch (e: any) {
                setStatus(`Error: ${e.message}`)
            } finally {
                setLoading(false)
            }
        }
        init()
    }, [sessionId])

    // Fetch mask for selected instance when frame changes
    useEffect(() => {
        if (selectedInstance !== null && currentFrame) {
            fetchInstanceMask(selectedInstance, currentFrame, kfIndex)
        } else if (promptPoints.length === 0) {
            setMaskOverlay(null)
        }
    }, [kfIndex, selectedInstance])

    // Clear prompt points when frame or mode changes
    useEffect(() => { setPromptPoints([]) }, [kfIndex, mode])

    // ── Canvas overlay sync & redraw ──────────────────────────
    const redrawCanvas = useCallback(() => {
        const canvas = overlayCanvasRef.current
        const img = imgRef.current
        if (!canvas || !img) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Sync canvas size to displayed image size
        // Use getBoundingClientRect like stac project's syncCanvas()
        const rect = img.getBoundingClientRect()
        const displayW = Math.round(rect.width)
        const displayH = Math.round(rect.height)
        if (displayW <= 0 || displayH <= 0) return

        if (canvas.width !== displayW || canvas.height !== displayH) {
            canvas.width = displayW
            canvas.height = displayH
        }
        // Sync CSS display size to match buffer (canvas is position:absolute, no CSS width/height)
        canvas.style.width = `${displayW}px`
        canvas.style.height = `${displayH}px`

        ctx.clearRect(0, 0, canvas.width, canvas.height)

        // Draw mask overlay (stretched to canvas = image displayed size)
        if (maskImageRef.current?.complete && maskImageRef.current.naturalWidth > 0) {
            ctx.save()
            ctx.globalAlpha = 0.55
            ctx.drawImage(maskImageRef.current, 0, 0, canvas.width, canvas.height)
            ctx.restore()
        }

        // Draw prompt points (natural coords → canvas coords)
        const natW = img.naturalWidth || 1
        const natH = img.naturalHeight || 1
        const scaleX = canvas.width / natW
        const scaleY = canvas.height / natH

        promptPoints.forEach((pt) => {
            const cx = pt.x * scaleX
            const cy = pt.y * scaleY

            ctx.beginPath()
            ctx.arc(cx, cy, 7, 0, Math.PI * 2)
            ctx.fillStyle = pt.label === 1 ? '#2ecc71' : '#e74c3c'
            ctx.fill()
            ctx.strokeStyle = '#fff'
            ctx.lineWidth = 2
            ctx.stroke()
        })
    }, [promptPoints, maskOverlay])

    // Redraw canvas whenever mask or points change
    useEffect(() => {
        // Load mask image into a hidden Image element for canvas drawing
        if (maskOverlay) {
            const img = new Image()
            img.onload = () => {
                maskImageRef.current = img
                redrawCanvas()
            }
            img.src = maskOverlay
        } else {
            maskImageRef.current = null
            redrawCanvas()
        }
    }, [maskOverlay, redrawCanvas])

    useEffect(() => { redrawCanvas() }, [promptPoints, maskOverlay, redrawCanvas])

    // Fetch mask for a specific instance on current frame
    const fetchInstanceMask = useCallback(async (instId: number, frame: string, frameSeqIdx?: number) => {
        try {
            // Use kf_index (sequential position) for NPZ lookup since masks use SAM3 sequential indices
            const params = new URLSearchParams({ frame })
            if (frameSeqIdx !== undefined) params.set('kf_index', String(frameSeqIdx))
            const res = await fetch(
                `/api/sessions/${sessionId}/segmentation/mask/${instId}?${params}`
            )
            if (!res.ok) { setMaskOverlay(null); return }
            const data = await res.json()
            if (data.mask_png) {
                setMaskOverlay(`data:image/png;base64,${data.mask_png}`)
            } else {
                setMaskOverlay(null)
            }
        } catch {
            setMaskOverlay(null)
        }
    }, [sessionId])

    // ── Manual click handler ──────────────────────────────────
    const handleImageClick = useCallback(async (e: React.MouseEvent<HTMLCanvasElement>, isPositive: boolean) => {
        e.preventDefault()
        // Suppress click if user was dragging (panning)
        if (wasDragging.current) {
            wasDragging.current = false
            return
        }
        if (mode !== 'manual' || !currentFrame) return
        if (!imgRef.current || !overlayCanvasRef.current) return

        const rect = overlayCanvasRef.current.getBoundingClientRect()
        const natW = imgRef.current.naturalWidth
        const natH = imgRef.current.naturalHeight
        const scaleX = natW / rect.width
        const scaleY = natH / rect.height
        const x = (e.clientX - rect.left) * scaleX
        const y = (e.clientY - rect.top) * scaleY

        // ── Edit existing instance (paint mode) ──
        if (selectedInstance !== null) {
            try {
                setStatus(`${isPositive ? 'Adding' : 'Removing'} mask area...`)
                const res = await fetch(`/api/segmentation/paint_mask`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        instance_id: selectedInstance,
                        kf_index: kfIndex,
                        x: x / natW,
                        y: y / natH,
                        radius: 15,
                        action: isPositive ? 'add' : 'remove'
                    })
                })
                if (res.ok) {
                    const data = await res.json()
                    if (data.mask_png) setMaskOverlay(`data:image/png;base64,${data.mask_png}`)
                    setStatus(`Mask updated (${isPositive ? 'added' : 'removed'} area)`)
                }
            } catch { setStatus('Error painting mask') }
            return
        }

        // ── New instance mode (SAM3 prompts) ──
        if (!stateId || loading) return

        // Hit-test: check if click is near an existing prompt dot (for removal)
        const HIT_RADIUS = 30  // pixels in natural image coords — generous for easy clicking
        const hitIdx = promptPoints.findIndex(pt => {
            const dx = pt.x - x
            const dy = pt.y - y
            return Math.sqrt(dx * dx + dy * dy) < HIT_RADIUS
        })

        if (hitIdx >= 0) {
            // Remove the clicked dot
            const updated = promptPoints.filter((_, i) => i !== hitIdx)
            setPromptPoints(updated)

            if (updated.length === 0) {
                setMaskOverlay(null)
                setHasPrompts(false)
                setStatus('All prompts removed.')
                return
            }

            // Re-send remaining points to SAM3
            try {
                setLoading(true)
                setStatus('Updating mask...')
                const res = await fetch(`/api/segmentation/add_prompt`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        state_id: stateId, session_id: sessionId,
                        frame_idx: currentFrameIdx, obj_id: 1,
                        // SAM3 expects relative coords (0-1), normalize by natural image size
                        points: updated.map(p => [p.x / natW, p.y / natH]),
                        labels: updated.map(p => p.label)
                    })
                })
                if (res.ok) {
                    const data = await res.json()
                    if (data.mask_png) setMaskOverlay(`data:image/png;base64,${data.mask_png}`)
                }
                setStatus('Prompt updated.')
            } catch { setStatus('Error updating mask') }
            finally { setLoading(false) }
            return
        }

        // No hit — add new point
        try {
            setLoading(true)
            setStatus(`Adding ${isPositive ? '✅ positive' : '❌ negative'} prompt...`)

            // Track the point locally
            const newPoints = [...promptPoints, { x, y, label: isPositive ? 1 : 0 }]
            setPromptPoints(newPoints)

            // Send ALL accumulated points to SAM3 (it uses cumulative prompts)
            const res = await fetch(`/api/segmentation/add_prompt`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    state_id: stateId, session_id: sessionId,
                    frame_idx: currentFrameIdx, obj_id: 1,
                    // SAM3 expects relative coords (0-1), normalize by natural image size
                    points: [[x / natW, y / natH]],
                    labels: [isPositive ? 1 : 0]
                })
            })
            if (!res.ok) throw new Error('Failed to add prompt')
            const data = await res.json()
            if (data.mask_png) setMaskOverlay(`data:image/png;base64,${data.mask_png}`)
            setHasPrompts(true)
            setStatus('Prompt applied. Add more or propagate.')
        } catch (e: any) {
            setStatus(`Error: ${e.message}`)
        } finally {
            setLoading(false)
        }
    }, [mode, stateId, sessionId, loading, currentFrame, currentFrameIdx, promptPoints, selectedInstance, kfIndex])

    // ── Hover handler: cursor feedback for removable dots ──────
    const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!imgRef.current || !overlayCanvasRef.current || promptPoints.length === 0) {
            if (overlayCanvasRef.current) overlayCanvasRef.current.style.cursor = 'crosshair'
            return
        }
        const rect = overlayCanvasRef.current.getBoundingClientRect()
        const natW = imgRef.current.naturalWidth
        const natH = imgRef.current.naturalHeight
        const x = (e.clientX - rect.left) * (natW / rect.width)
        const y = (e.clientY - rect.top) * (natH / rect.height)
        const HIT_RADIUS = 30
        const isOverDot = promptPoints.some(pt =>
            Math.sqrt((pt.x - x) ** 2 + (pt.y - y) ** 2) < HIT_RADIUS
        )
        overlayCanvasRef.current.style.cursor = isOverDot ? 'pointer' : 'crosshair'
    }, [promptPoints])

    // ── Text prompt handler ──────────────────────────────────
    const handleTextPrompt = useCallback(async () => {
        if (!stateId || !textPrompt.trim() || loading) return
        try {
            setLoading(true)
            setStatus(`Text prompt: "${textPrompt}"...`)
            const res = await fetch(`/api/segmentation/text_prompt`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    state_id: stateId, frame_idx: currentFrameIdx,
                    text: textPrompt.trim(), obj_id: 1
                })
            })
            if (!res.ok) throw new Error('Text prompt failed')
            const data = await res.json()
            if (data.mask_png) setMaskOverlay(`data:image/png;base64,${data.mask_png}`)
            setHasPrompts(true)
            setStatus('Text prompt applied. Add more or propagate.')
        } catch (e: any) {
            setStatus(`Error: ${e.message}`)
        } finally {
            setLoading(false)
        }
    }, [stateId, textPrompt, loading, currentFrameIdx])

    // ── Propagate handler (SSE streaming) ─────────────────
    const [propagationPct, setPropagationPct] = useState(-1)  // -1 = not propagating

    const handlePropagate = useCallback(async () => {
        if (!stateId) return
        const name = labelName.trim() || 'manual_object'
        let receivedDone = false
        try {
            setLoading(true)
            setPropagationPct(0)
            setStatus(`Propagating "${name}"...`)

            const res = await fetch(`/api/segmentation/propagate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ state_id: stateId, session_id: sessionId, label_name: name })
            })

            if (!res.ok) throw new Error('Propagation failed')
            if (!res.body) throw new Error('No response body')

            const reader = res.body.getReader()
            const decoder = new TextDecoder()
            let buffer = ''

            const processLines = async (lines: string[]) => {
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i]
                    if (line.startsWith('event: ')) {
                        const eventType = line.slice(7).trim()
                        // Look ahead for the data line
                        const nextLine = i + 1 < lines.length ? lines[i + 1] : null
                        if (nextLine && nextLine.startsWith('data: ')) {
                            i++ // consume the data line
                            let data: any
                            try {
                                data = JSON.parse(nextLine.slice(6))
                            } catch {
                                continue
                            }

                            if (eventType === 'progress') {
                                setPropagationPct(data.pct || 0)
                                setKfIndex(data.frame)
                                setStatus(`Propagating "${name}": frame ${data.frame + 1}/${data.total} (${data.pct}%)`)
                                if (data.mask_png) {
                                    setMaskOverlay(`data:image/png;base64,${data.mask_png}`)
                                }
                            } else if (eventType === 'saving') {
                                setStatus(data.status || 'Saving...')
                            } else if (eventType === 'done') {
                                receivedDone = true
                                setPropagationPct(-1)
                                setHasPrompts(false)
                                setMaskOverlay(null)
                                setLabelName('')
                                setStatus(`✅ "${name}" segmented successfully!`)

                                // Reload instances from server
                                try {
                                    const segRes = await fetch(`/api/sessions/${sessionId}/segmentation`)
                                    if (segRes.ok) {
                                        const segData = await segRes.json()
                                        setInstances((segData.instances || []).map((inst: any, i: number) => ({
                                            id: inst.id,
                                            label: inst.label || `Object ${inst.id}`,
                                            color: inst.color || INSTANCE_COLORS[i % INSTANCE_COLORS.length],
                                            total_points: inst.total_points || 0,
                                            excluded: inst.excluded || false,
                                        })))
                                    }
                                } catch (fetchErr) {
                                    console.error('Failed to reload instances:', fetchErr)
                                }
                            } else if (eventType === 'error') {
                                throw new Error(data.message || 'Propagation error')
                            }
                        }
                    }
                }
            }

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split('\n')
                buffer = lines.pop() || ''
                await processLines(lines)
            }

            // Process any remaining buffer after stream ends
            if (buffer.trim()) {
                await processLines(buffer.split('\n'))
            }

            // Safety: if stream ended but we never got 'done', force cleanup
            if (!receivedDone) {
                console.warn('[SegManager] Stream ended without done event, forcing cleanup')
                setPropagationPct(-1)
                setHasPrompts(false)
                setMaskOverlay(null)
                setLabelName('')
                setStatus(`✅ "${name}" propagation finished`)
                // Reload instances
                try {
                    const segRes = await fetch(`/api/sessions/${sessionId}/segmentation`)
                    if (segRes.ok) {
                        const segData = await segRes.json()
                        setInstances((segData.instances || []).map((inst: any, i: number) => ({
                            id: inst.id,
                            label: inst.label || `Object ${inst.id}`,
                            color: inst.color || INSTANCE_COLORS[i % INSTANCE_COLORS.length],
                            total_points: inst.total_points || 0,
                            excluded: inst.excluded || false,
                        })))
                    }
                } catch { /* silent */ }
            }
        } catch (e: any) {
            setStatus(`Error: ${e.message}`)
            setPropagationPct(-1)
        } finally {
            setLoading(false)
        }
    }, [stateId, sessionId, labelName])

    // ── Auto segmentation ──────────────────────────────────
    const handleAutoSegment = useCallback(async () => {
        try {
            setLoading(true)
            setStatus('🤖 Running VLM analysis + auto-segmentation...')
            const res = await fetch(`/api/segmentation/auto/${sessionId}`, { method: 'POST' })
            if (!res.ok) throw new Error('Auto-segmentation failed')
            const data = await res.json()
            setStatus(`✅ Auto-segmentation complete! ${(data.instances || []).length} instances`)

            // Reload instances
            const segRes = await fetch(`/api/sessions/${sessionId}/segmentation`)
            if (segRes.ok) {
                const segData = await segRes.json()
                setInstances((segData.instances || []).map((inst: any, i: number) => ({
                    id: inst.id,
                    label: inst.label || `Object ${inst.id}`,
                    color: inst.color || INSTANCE_COLORS[i % INSTANCE_COLORS.length],
                    total_points: inst.total_points || 0,
                    excluded: inst.excluded || false,
                })))
            }
        } catch (e: any) {
            setStatus(`Error: ${e.message}`)
        } finally {
            setLoading(false)
        }
    }, [sessionId])

    // ── Rename handler ──────────────────────────────────
    const handleRename = useCallback(async (instId: number, newLabel: string) => {
        try {
            await fetch(`/api/segmentation/rename`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, instance_id: instId, label: newLabel })
            })
            setInstances(prev => prev.map(i => i.id === instId ? { ...i, label: newLabel } : i))
            onUpdate?.()
        } catch { /* silent */ }
    }, [sessionId, onUpdate])

    // ── Exclusion toggle ──────────────────────────────────
    const handleToggleExclude = useCallback(async (instId: number, excluded: boolean) => {
        try {
            await fetch(`/api/segmentation/rename`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, instance_id: instId, excluded })
            })
            setInstances(prev => prev.map(i => i.id === instId ? { ...i, excluded } : i))
            onUpdate?.()
        } catch { /* silent */ }
    }, [sessionId, onUpdate])

    // ── Delete instance ──────────────────────────────────
    const handleDelete = useCallback(async (instId: number, label: string) => {
        const ok = await confirmDanger(`Delete "${label}"? This removes all masks and 3D data.`, 'Delete Instance')
        if (!ok) return
        try {
            setStatus(`Deleting "${label}"...`)
            const res = await fetch(`/api/segmentation/delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, instance_id: instId })
            })
            if (!res.ok) throw new Error('Delete failed')
            setInstances(prev => prev.filter(i => i.id !== instId))
            if (selectedInstance === instId) {
                setSelectedInstance(null)
                setMaskOverlay(null)
            }
            setStatus(`✅ "${label}" deleted`)
            onUpdate?.()
        } catch (e: any) {
            setStatus(`Error: ${e.message}`)
        }
    }, [sessionId, selectedInstance, onUpdate])

    // ── Zoom/Pan handlers ──────────────────────────────────
    const handleWheel = useCallback((e: React.WheelEvent) => {
        e.preventDefault()
        setZoom(z => Math.max(0.5, Math.min(8, z + (e.deltaY > 0 ? -0.2 : 0.2))))
    }, [])

    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        if (e.button === 1 || (e.shiftKey && e.button === 0)) {
            e.preventDefault()
            setIsPanning(true)
            panStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y }
        }
        // Track mousedown position for drag detection (suppress click after drag)
        mouseDownPos.current = { x: e.clientX, y: e.clientY }
    }, [pan])

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
        if (!isPanning) return
        setPan({
            x: panStart.current.panX + (e.clientX - panStart.current.x),
            y: panStart.current.panY + (e.clientY - panStart.current.y),
        })
    }, [isPanning])

    const handleMouseUp = useCallback((e: React.MouseEvent | React.SyntheticEvent) => {
        // Detect if this was a drag (moved > 5px from mousedown)
        const nativeEvent = 'clientX' in e ? e as React.MouseEvent : null
        if (nativeEvent && mouseDownPos.current) {
            const dx = nativeEvent.clientX - mouseDownPos.current.x
            const dy = nativeEvent.clientY - mouseDownPos.current.y
            wasDragging.current = Math.sqrt(dx * dx + dy * dy) > 5
        }
        setIsPanning(false)
    }, [])

    const resetView = useCallback(() => {
        setZoom(1)
        setPan({ x: 0, y: 0 })
    }, [])

    // ── Render ──────────────────────────────────────────────
    return (
        <div className="admin-overlay">
            <div className="admin-panel seg-manager">
                {/* Header */}
                <div className="admin-header">
                    <h2>🏷️ Segmentation Manager — {sessionId}</h2>
                    <button className="admin-close" onClick={onClose} disabled={loading}>✕</button>
                </div>

                <div className="seg-manager-body">
                    {/* ── Left: Instance List ── */}
                    <div className="seg-sidebar">
                        <div className="seg-sidebar-header">
                            <span>Instances ({instances.length})</span>
                        </div>

                        <div className="seg-instance-list">
                            {instances.length === 0 && (
                                <div className="seg-empty">No instances yet. Add prompts and propagate.</div>
                            )}
                            {instances.map((inst) => (
                                <div
                                    key={inst.id}
                                    className={`seg-instance-item ${selectedInstance === inst.id ? 'selected' : ''} ${inst.excluded ? 'excluded' : ''}`}
                                    onClick={() => {
                                        const newSel = inst.id === selectedInstance ? null : inst.id
                                        setSelectedInstance(newSel)
                                        if (newSel !== null && currentFrame) {
                                            fetchInstanceMask(newSel, currentFrame, kfIndex)
                                        } else {
                                            setMaskOverlay(null)
                                        }
                                    }}
                                >
                                    <span className="seg-inst-color" style={{ background: inst.color }} />
                                    <input
                                        className="seg-inst-name"
                                        value={inst.label}
                                        onChange={e => {
                                            const val = e.target.value
                                            setInstances(prev => prev.map(i => i.id === inst.id ? { ...i, label: val } : i))
                                        }}
                                        onBlur={e => handleRename(inst.id, e.target.value)}
                                        onClick={e => e.stopPropagation()}
                                    />
                                    <span className="seg-inst-pts">{inst.total_points.toLocaleString()}</span>
                                    <div className="seg-inst-actions">
                                        <button
                                            className={`seg-inst-btn seg-inst-edit ${selectedInstance === inst.id ? 'active' : ''}`}
                                            title="Edit mask (paint mode)"
                                            onClick={e => {
                                                e.stopPropagation()
                                                const newSel = inst.id === selectedInstance ? null : inst.id
                                                setSelectedInstance(newSel)
                                                if (newSel !== null && currentFrame) {
                                                    fetchInstanceMask(newSel, currentFrame, kfIndex)
                                                    setStatus(`Editing "${inst.label}" — Left click = add, Right click = remove`)
                                                } else {
                                                    setMaskOverlay(null)
                                                }
                                            }}
                                        >✏️</button>
                                        <button
                                            className={`seg-inst-btn seg-inst-excl ${inst.excluded ? 'on' : ''}`}
                                            title={inst.excluded ? 'Include in cloud' : 'Exclude from cloud'}
                                            onClick={e => { e.stopPropagation(); handleToggleExclude(inst.id, !inst.excluded) }}
                                        >{inst.excluded ? '⊘' : '👁'}</button>
                                        <button
                                            className="seg-inst-btn seg-inst-del"
                                            title="Delete instance"
                                            onClick={e => { e.stopPropagation(); handleDelete(inst.id, inst.label) }}
                                        >✕</button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* ── Right: Viewer ── */}
                    <div className="seg-viewer">
                        {/* Status bar */}
                        <div className="seg-status-bar">
                            <span className={`seg-status-dot ${loading ? 'busy' : 'ready'}`} />
                            <span className="seg-status-text">{status}</span>
                        </div>

                        {/* Progress bar (visible during propagation) */}
                        {propagationPct >= 0 && (
                            <div className="seg-progress-bar">
                                <div className="seg-progress-fill" style={{ width: `${propagationPct}%` }} />
                                <span className="seg-progress-text">{propagationPct}%</span>
                            </div>
                        )}

                        {/* Mode toolbar */}
                        <div className="seg-toolbar">
                            <div className="seg-mode-group">
                                <button className={`seg-mode-btn ${mode === 'manual' ? 'active' : ''}`}
                                    onClick={() => setMode('manual')} disabled={loading}>🖱️ Manual</button>
                                <button className={`seg-mode-btn ${mode === 'text' ? 'active' : ''}`}
                                    onClick={() => setMode('text')} disabled={loading}>📝 Text</button>
                                <button className={`seg-mode-btn ${mode === 'auto' ? 'active' : ''}`}
                                    onClick={() => setMode('auto')} disabled={loading}>🤖 Auto</button>
                            </div>

                            <div className="seg-nav-group">
                                <button className="seg-nav-btn" disabled={kfIndex <= 0}
                                    onClick={() => setKfIndex(i => i - 1)}>◀</button>
                                <span className="seg-nav-label">
                                    KF {kfIndex + 1}/{keyframes.length}
                                    {currentFrame && <span className="seg-nav-filename">{currentFrame}</span>}
                                </span>
                                <button className="seg-nav-btn" disabled={kfIndex >= keyframes.length - 1}
                                    onClick={() => setKfIndex(i => i + 1)}>▶</button>
                            </div>

                            <div className="seg-zoom-group">
                                <button className="seg-nav-btn" onClick={() => setZoom(z => Math.min(8, z + 0.5))}>+</button>
                                <span className="seg-zoom-label">{Math.round(zoom * 100)}%</span>
                                <button className="seg-nav-btn" onClick={() => setZoom(z => Math.max(0.5, z - 0.5))}>−</button>
                                <button className="seg-nav-btn" onClick={resetView} title="Reset">⟲</button>
                            </div>
                        </div>

                        {/* Text prompt bar (when in text mode) */}
                        {mode === 'text' && (
                            <div className="seg-text-bar">
                                <input
                                    className="seg-text-input"
                                    placeholder='Describe the object (e.g. "the red sofa")'
                                    value={textPrompt}
                                    onChange={e => setTextPrompt(e.target.value)}
                                    onKeyDown={e => e.key === 'Enter' && handleTextPrompt()}
                                    disabled={loading}
                                />
                                <button className="admin-save-btn" onClick={handleTextPrompt}
                                    disabled={loading || !textPrompt.trim()}>Segment</button>
                            </div>
                        )}

                        {/* Auto mode button */}
                        {mode === 'auto' && (
                            <div className="seg-text-bar">
                                <span className="seg-auto-desc">
                                    VLM will analyze all keyframes, detect objects, and segment them automatically.
                                </span>
                                <button className="admin-save-btn" onClick={handleAutoSegment}
                                    disabled={loading}>🤖 Start Auto-Segmentation</button>
                            </div>
                        )}

                        {/* Image canvas with zoom/pan */}
                        <div
                            ref={canvasRef}
                            className={`seg-canvas ${loading ? 'loading' : ''} ${mode === 'manual' ? 'crosshair' : ''}`}
                            onWheel={handleWheel}
                            onMouseDown={handleMouseDown}
                            onMouseMove={handleMouseMove}
                            onMouseUp={handleMouseUp}
                            onMouseLeave={handleMouseUp}
                        >
                            {currentFrame ? (
                                <div
                                    className="seg-transform-container"
                                    style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}
                                >
                                    <div className="seg-image-wrapper">
                                        <img
                                            ref={imgRef}
                                            src={`/api/sessions/${sessionId}/frames/${currentFrame}`}
                                            className="seg-image"
                                            alt={`Keyframe ${currentFrame}`}
                                            draggable={false}
                                            onLoad={() => redrawCanvas()}
                                        />
                                        <canvas
                                            ref={overlayCanvasRef}
                                            className="seg-overlay-canvas"
                                            onClick={(e) => handleImageClick(e, true)}
                                            onContextMenu={(e) => handleImageClick(e, false)}
                                            onMouseMove={handleCanvasMouseMove}
                                        />
                                    </div>
                                </div>
                            ) : (
                                <div className="seg-placeholder">No keyframes loaded</div>
                            )}
                        </div>

                        {/* Footer */}
                        <div className="seg-footer">
                            <span className="seg-hint">
                                {mode === 'manual' && '🖱️ Left = Positive | Right = Negative | Scroll = Zoom | Shift+Drag = Pan'}
                                {mode === 'text' && '📝 Type a description and press Enter or Segment'}
                                {mode === 'auto' && '🤖 Auto mode uses VLM + Segmentation'}
                            </span>
                            <div className="seg-actions">
                                <input
                                    className="seg-label-input"
                                    type="text"
                                    placeholder="Label name..."
                                    value={labelName}
                                    onChange={e => setLabelName(e.target.value)}
                                    disabled={loading}
                                />
                                <button
                                    style={{ background: '#555', color: '#fff', border: 'none', borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 13 }}
                                    onClick={() => {
                                        setPromptPoints([])
                                        setMaskOverlay(null)
                                        setHasPrompts(false)
                                        setStatus('Prompts cleared.')
                                    }}
                                    disabled={loading || promptPoints.length === 0}
                                >✕ Clear</button>
                                <button className="admin-save-btn" onClick={handlePropagate}
                                    disabled={loading || !stateId || !hasPrompts}>▶ Propagate</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {dialogElement}
        </div>
    )
}
