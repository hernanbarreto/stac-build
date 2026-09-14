import { useState, useRef, useEffect, useCallback } from 'react'
import { X, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Maximize, Check, Pencil, Eye, EyeOff, Plus, Play, Trash2, Brain, MousePointer, Type, Bot, Crosshair } from 'lucide-react'
import { useConfirmDialog } from './ConfirmDialog'
import { Dialog } from './ui/Dialog'
import { Button } from './ui/Button'
import { IconButton } from './ui/IconButton'
import { Input, SegmentedControl, Checkbox } from './ui/Field'
import { Badge, ColorDot } from './ui/Badge'
import { Progress } from './ui/Progress'
import { Tooltip } from './ui/Tooltip'
import { Sep } from './ui/Panel'
import { EmptyState } from './ui/EmptyState'
import { categoricalPalette, tokenColor, VP } from './viewport/palette'
import { getT, useFmt, useT } from '../i18n'

/** Translator for callbacks (active language read at call time). */
const tt = (key: string, vars?: Record<string, string | number | null | undefined>) => getT()(key, vars)

interface InstanceInfo {
    id: number
    label: string
    color: string
    total_points: number
    excluded?: boolean
}

interface Props {
    sessionId: string
    onClose: (dirty: boolean) => void
    onSuccess?: (instances: any[]) => void
    onUpdate?: () => void
}

type SegMode = 'manual' | 'text' | 'auto'

// New instances take the categorical palette from the design tokens (--cat-*)
const instanceColors = () => categoricalPalette()

export default function SegmentationManager({ sessionId, onClose, onUpdate }: Props) {
    const { confirmDanger, dialogElement } = useConfirmDialog()
    // State
    const [loading, setLoading] = useState(true)
    const [stateId, setStateId] = useState<string | null>(null)
    const [keyframes, setKeyframes] = useState<string[]>([])
    const [kfIndex, setKfIndex] = useState(0)
    const [status, setStatus] = useState(() => tt('seg.initializing'))
    const [mode, setMode] = useState<SegMode>('manual')
    const [instances, setInstances] = useState<InstanceInfo[]>([])
    const [selectedInstance, setSelectedInstance] = useState<number | null>(null)
    const [maskOverlay, setMaskOverlay] = useState<string | null>(null)
    const [hasPrompts, setHasPrompts] = useState(false)
    const [labelName, setLabelName] = useState('')
    const [textPrompt, setTextPrompt] = useState('')
    const [promptPoints, setPromptPoints] = useState<{ x: number, y: number, label: number }[]>([])
    // Prompt/mask caches are keyed by "<frame>:<obj_id>" — NOT by frame alone.
    // Two objects edited on the same frame must never share an entry, or the
    // previous object's points leak into the next object's payload (bug 4).
    const promptPointsMapRef = useRef<Map<string, { x: number, y: number, label: number }[]>>(new Map())
    const maskCacheRef = useRef<Map<string, string>>(new Map())
    const [selectedFrames, setSelectedFrames] = useState<Set<number>>(new Set())

    // ── Multi-object queue ───────────────────────────────────────
    // Objects already "added" this batch (prompts live in the SAM3 session under
    // their obj_id). The object currently being clicked uses currentObjIdRef. On
    // "Propagate all" every queued obj_id + the in-edit one is tracked at once and
    // saved with its own label. obj_ids start ABOVE the max existing instance id so
    // _save_masks' upsert never overwrites a previously-saved object.
    const [pendingObjects, setPendingObjects] = useState<{ objId: number, name: string, color: string }[]>([])
    const currentObjIdRef = useRef(1)
    const instancesRef = useRef<InstanceInfo[]>([])

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
    const isDirty = useRef(false)
    const propagatingRef = useRef(false)
    const promptingRef = useRef(false)
    const stateIdRef = useRef<string | null>(null)

    // Keep ref in sync with state for cleanup
    useEffect(() => { stateIdRef.current = stateId }, [stateId])

    // Keep instancesRef current and, while no object is mid-edit/queued, park the
    // next obj_id above the highest saved id so a new batch never collides with
    // (and overwrites) an existing instance.
    useEffect(() => {
        instancesRef.current = instances
        if (pendingObjects.length === 0 && promptPoints.length === 0) {
            currentObjIdRef.current = instances.reduce((m, i) => Math.max(m, i.id), 0) + 1
        }
    }, [instances, pendingObjects.length, promptPoints.length])

    // Cleanup SAM3 session on unmount (free VRAM)
    useEffect(() => {
        return () => {
            const sid = stateIdRef.current
            if (sid) {
                fetch('/api/segmentation/reset_session', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ state_id: sid })
                }).catch(() => { })  // fire-and-forget
            }
        }
    }, [])

    const currentFrame = keyframes[kfIndex] || null

    // Keep the active thumbnail visible in the carousel when the viewer
    // navigates (thumbnail click, / buttons, or propagate progress).
    const thumbRefs = useRef<(HTMLDivElement | null)[]>([])
    useEffect(() => {
        thumbRefs.current[kfIndex]?.scrollIntoView({
            behavior: 'smooth', block: 'nearest', inline: 'center',
        })
    }, [kfIndex])
    // SAM3 now loads only keyframes sequentially (via temp dir).
    // The SAM3 frame_idx is the position in the sorted keyframes list = kfIndex.
    const currentFrameIdx = kfIndex
    // Live mirror of kfIndex for async handlers: a slow add_prompt response
    // must not paint its mask over a DIFFERENT frame/object the user has
    // navigated to meanwhile (stale-response race).
    const kfIndexRef = useRef(kfIndex)
    useEffect(() => { kfIndexRef.current = kfIndex }, [kfIndex])

    // ── Init ──────────────────────────────────────────────────
    useEffect(() => {
        if (initRef.current) return
        initRef.current = true

        const init = async () => {
            try {
                setLoading(true)

                // Load existing instances
                setStatus(tt('seg.loadingExisting'))
                const segRes = await fetch(`/api/sessions/${sessionId}/segmentation`)
                if (segRes.ok) {
                    const segData = await segRes.json()
                    setInstances((segData.instances || []).map((inst: any, i: number) => ({
                        id: inst.id,
                        label: inst.label || tt('seg.objectFallback', { id: inst.id }),
                        color: inst.color || instanceColors()[i % instanceColors().length],
                        total_points: inst.total_points || 0,
                        excluded: inst.excluded || false,
                    })))
                }

                // Load keyframes
                setStatus(tt('seg.loadingKeyframes'))
                const kfRes = await fetch(`/api/sessions/${sessionId}/keyframes`)
                if (!kfRes.ok) throw new Error(tt('seg.keyframesFailed'))
                const kfData = await kfRes.json()
                const kfList: string[] = kfData.keyframes || []
                if (kfList.length === 0) throw new Error(tt('seg.noKeyframes'))
                setKeyframes(kfList)
                setKfIndex(0)
                setSelectedFrames(new Set(kfList.map((_, i) => i)))

                // Fire-and-forget: kick off SAM3 init (server returns 202 immediately)
                setStatus(tt('seg.initModel', { n: kfList.length }))
                fetch(`/api/segmentation/start_session/${sessionId}`, { method: 'POST' }).catch(() => { })

                // Poll lightweight status endpoint every 2s until ready
                const startTime = Date.now()
                const result = await new Promise<any>((resolve, reject) => {
                    const poll = setInterval(async () => {
                        try {
                            const elapsed = Math.round((Date.now() - startTime) / 1000)
                            const dots = '.'.repeat((elapsed % 3) + 1)
                            if (elapsed < 10) {
                                setStatus(tt('seg.initModel', { n: kfList.length }) + dots)
                            } else {
                                setStatus(tt('seg.loadingModel', { s: elapsed }) + dots)
                            }

                            const r = await fetch(`/api/segmentation/init_status/${sessionId}`)
                            if (!r.ok) return
                            const data = await r.json()

                            if (data.status === 'ready') {
                                clearInterval(poll)
                                resolve(data)
                            } else if (data.status === 'error') {
                                clearInterval(poll)
                                reject(new Error(data.message || tt('seg.initFailed')))
                            }
                        } catch { /* poll retry */ }
                    }, 2000)
                })

                setStateId(result.state_id)
                setStatus(tt('seg.ready', { n: kfList.length }))
            } catch (e: any) {
                setStatus(tt('seg.error', { detail: e.message }))
            } finally {
                setLoading(false)
            }
        }
        init()
    }, [sessionId])

    // Cache key: prompts/masks belong to (frame, object-in-edit), never to the
    // frame alone — see promptPointsMapRef above (bug 4).
    const pKey = (frame: number) => `${frame}:${currentObjIdRef.current}`

    // Save current frame's prompts before navigating, restore target frame's prompts
    const prevKfIndexRef = useRef(kfIndex)
    useEffect(() => {
        const prev = prevKfIndexRef.current
        // Save current frame's prompt points (under this object's key)
        if (promptPoints.length > 0) {
            promptPointsMapRef.current.set(pKey(prev), [...promptPoints])
        }
        prevKfIndexRef.current = kfIndex

        if (selectedInstance !== null && currentFrame) {
            fetchInstanceMask(selectedInstance, currentFrame, kfIndex)
            return
        }

        // Restore target frame's prompts & cached mask (this object only)
        const savedPoints = promptPointsMapRef.current.get(pKey(kfIndex)) || []
        setPromptPoints(savedPoints)
        const cachedMask = maskCacheRef.current.get(pKey(kfIndex))
        if (cachedMask) {
            setMaskOverlay(cachedMask)
        } else if (savedPoints.length === 0) {
            setMaskOverlay(null)
        }
    }, [kfIndex, selectedInstance])

    // Changing the ACTIVE object (selecting an existing instance to edit, or
    // deselecting it) must clear the previous object's on-screen prompts: its
    // dots would otherwise stay live and be re-sent under the new obj_id
    // (the bug-4 contamination). The confirmed masks of saved instances stay
    // visible through fetchInstanceMask; only in-edit prompt state resets.
    const prevSelectedInstanceRef = useRef(selectedInstance)
    useEffect(() => {
        if (prevSelectedInstanceRef.current !== selectedInstance) {
            prevSelectedInstanceRef.current = selectedInstance
            setPromptPoints([])
            setHasPrompts(false)
        }
    }, [selectedInstance])

    // Clear prompt points only when mode changes (not frame)
    useEffect(() => { setPromptPoints([]); promptPointsMapRef.current.clear(); maskCacheRef.current.clear() }, [mode])

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

        // Draw prompt points (natural coords  canvas coords)
        const natW = img.naturalWidth || 1
        const natH = img.naturalHeight || 1
        const scaleX = canvas.width / natW
        const scaleY = canvas.height / natH

        promptPoints.forEach((pt) => {
            const cx = pt.x * scaleX
            const cy = pt.y * scaleY

            ctx.beginPath()
            ctx.arc(cx, cy, 7, 0, Math.PI * 2)
            ctx.fillStyle = pt.label === 1 ? tokenColor(VP.ok) : tokenColor(VP.err)
            ctx.fill()
            ctx.strokeStyle = tokenColor(VP.light)
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

    // ── Clear the CURRENT object only (footer button + Esc) ─────
    // Never touches the queue, saved instances or the frame selection.
    const clearCurrentObject = useCallback(async () => {
        const objId = currentObjIdRef.current
        setPromptPoints([])
        setMaskOverlay(null)
        setHasPrompts(false)
        setLabelName('')
        for (const k of [...promptPointsMapRef.current.keys()]) {
            if (k.endsWith(`:${objId}`)) promptPointsMapRef.current.delete(k)
        }
        for (const k of [...maskCacheRef.current.keys()]) {
            if (k.endsWith(`:${objId}`)) maskCacheRef.current.delete(k)
        }
        if (stateId) {
            try {
                setLoading(true)
                setStatus(tt('seg.clearingObject'))
                const res = await fetch('/api/segmentation/clear_prompts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ state_id: stateId, obj_id: objId })
                })
                setStatus(res.ok ? tt('seg.objectCleared') : tt('seg.clearBackendWarning'))
            } catch { setStatus(tt('seg.errorClearing')) }
            finally { setLoading(false) }
        } else {
            setStatus(tt('seg.objectCleared'))
        }
    }, [stateId])

    // ── Undo the last prompt point (Ctrl/Cmd+Z) ─────────────────
    const undoLastPoint = useCallback(async () => {
        if (selectedInstance !== null || promptPoints.length === 0 || promptingRef.current) return
        const updated = promptPoints.slice(0, -1)
        setPromptPoints(updated)
        promptPointsMapRef.current.set(pKey(kfIndex), updated)
        if (updated.length === 0) {
            setMaskOverlay(null)
            setHasPrompts(false)
            maskCacheRef.current.delete(pKey(kfIndex))
            setStatus(tt('seg.undoneEmpty'))
            return
        }
        const img = imgRef.current
        if (!stateId || !img) return
        promptingRef.current = true
        try {
            setLoading(true)
            setStatus(tt('seg.undoing'))
            const res = await fetch('/api/segmentation/add_prompt', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    state_id: stateId, session_id: sessionId,
                    frame_idx: currentFrameIdx, obj_id: currentObjIdRef.current,
                    points: updated.map(p => [p.x / img.naturalWidth, p.y / img.naturalHeight]),
                    labels: updated.map(p => p.label)
                })
            })
            if (res.ok) {
                const data = await res.json()
                if (data.mask_png) {
                    const u = `data:image/png;base64,${data.mask_png}`
                    setMaskOverlay(u)
                    maskCacheRef.current.set(pKey(kfIndex), u)
                }
            }
            setStatus(tt('seg.undone'))
        } catch { setStatus(tt('seg.errorUndo')) }
        finally { setLoading(false); promptingRef.current = false }
    }, [promptPoints, selectedInstance, stateId, sessionId, kfIndex, currentFrameIdx])

    // ── Keyboard shortcuts: / navigate frames, Esc cancels the current
    // object's prompts, Ctrl/Cmd+Z undoes the last point. Ignored while
    // typing in inputs. ──
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            const t = e.target as HTMLElement | null
            if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
            if (e.key === 'ArrowLeft') {
                e.preventDefault()
                setKfIndex(i => Math.max(0, i - 1))
            } else if (e.key === 'ArrowRight') {
                e.preventDefault()
                setKfIndex(i => Math.min(keyframes.length - 1, i + 1))
            } else if (e.key === 'Escape') {
                if (hasPrompts) { e.preventDefault(); clearCurrentObject() }
            } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
                e.preventDefault()
                undoLastPoint()
            }
        }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    }, [keyframes.length, hasPrompts, clearCurrentObject, undoLastPoint])

    // ── Warn before leaving the page with unsaved segmentation work ──
    useEffect(() => {
        const dirty = hasPrompts || pendingObjects.length > 0
        if (!dirty) return
        const onBeforeUnload = (e: BeforeUnloadEvent) => {
            e.preventDefault()
            e.returnValue = ''
        }
        window.addEventListener('beforeunload', onBeforeUnload)
        return () => window.removeEventListener('beforeunload', onBeforeUnload)
    }, [hasPrompts, pendingObjects.length])

    // ── Manual click handler ──────────────────────────────────
    const lastClickTime = useRef(0)
    const handleImageClick = useCallback(async (e: React.MouseEvent<HTMLCanvasElement>, isPositive: boolean) => {
        e.preventDefault()
        e.stopPropagation()
        // Debounce: ignore clicks within 300ms of each other
        const now = Date.now()
        if (now - lastClickTime.current < 300) return
        lastClickTime.current = now
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
                setStatus(isPositive ? tt('seg.addingMaskArea') : tt('seg.removingMaskArea'))
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
                    setStatus(isPositive ? tt('seg.maskAreaAdded') : tt('seg.maskAreaRemoved'))
                    isDirty.current = true
                }
            } catch { setStatus(tt('seg.errorPainting')) }
            return
        }

        // ── New instance mode (SAM3 prompts) ──
        if (!stateId || loading || promptingRef.current) return
        promptingRef.current = true

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
                setStatus(tt('seg.allPromptsRemoved'))
                // Must release the click lock here too: this early return removed
                // the last dot, and without resetting it promptingRef stays true
                // forever  every later click (this image or any other) is ignored.
                promptingRef.current = false
                return
            }

            // Re-send remaining points to SAM3
            try {
                setLoading(true)
                setStatus(tt('seg.updatingMask'))
                const res = await fetch(`/api/segmentation/add_prompt`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        state_id: stateId, session_id: sessionId,
                        frame_idx: currentFrameIdx, obj_id: currentObjIdRef.current,
                        // SAM3 expects relative coords (0-1), normalize by natural image size
                        points: updated.map(p => [p.x / natW, p.y / natH]),
                        labels: updated.map(p => p.label)
                    })
                })
                if (res.ok) {
                    const data = await res.json()
                    if (data.mask_png) setMaskOverlay(`data:image/png;base64,${data.mask_png}`)
                }
                setStatus(tt('seg.promptUpdated'))
            } catch { setStatus(tt('seg.errorUpdatingMask')) }
            finally { setLoading(false); promptingRef.current = false }
            return
        }

        // No hit — add new point
        try {
            setLoading(true)
            setStatus(isPositive ? tt('seg.addingPositive') : tt('seg.addingNegative'))

            // Snapshot the send context: if the user navigates to another
            // frame or object while this request is in flight, the response
            // is STALE and must not repaint the new context (race guard).
            const sentFrame = kfIndex
            const sentObj = currentObjIdRef.current

            // Track the point locally
            const newPoints = [...promptPoints, { x, y, label: isPositive ? 1 : 0 }]
            setPromptPoints(newPoints)

            // Send the FULL accumulated point set to SAM3, not just the new
            // click. SAM3's add_new_points_or_box defaults to
            // ``clear_old_points=True`` (see vendor/sam3/sam3/model/
            // sam3_tracking_predictor.py), so each call replaces prior
            // prompts for this (frame, obj_id). Sending only the latest point
            // is what made compound-object segmentations regress to just the
            // last click — the mask "forgot" everything except the most
            // recent input. The remove-point branch above already does this
            // correctly; we mirror it here.
            const res = await fetch(`/api/segmentation/add_prompt`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    state_id: stateId, session_id: sessionId,
                    frame_idx: currentFrameIdx, obj_id: currentObjIdRef.current,
                    points: newPoints.map(p => [p.x / natW, p.y / natH]),
                    labels: newPoints.map(p => p.label)
                }),
            })
            if (!res.ok) throw new Error(tt('seg.addPromptFailed'))
            const data = await res.json()
            const stale = kfIndexRef.current !== sentFrame
                || currentObjIdRef.current !== sentObj
            if (data.mask_png) {
                const maskUrl = `data:image/png;base64,${data.mask_png}`
                // Cache under the ORIGINAL context either way; only paint the
                // overlay if the user is still looking at that context.
                maskCacheRef.current.set(`${sentFrame}:${sentObj}`, maskUrl)
                if (!stale) setMaskOverlay(maskUrl)
            }
            promptPointsMapRef.current.set(`${sentFrame}:${sentObj}`, newPoints)
            setHasPrompts(true)
            setStatus(stale ? tt('seg.promptAppliedPrevious') : tt('seg.promptApplied'))
        } catch (e: any) {
            setStatus(tt('seg.error', { detail: e.message }))
        } finally {
            setLoading(false)
            promptingRef.current = false
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
            setStatus(tt('seg.textPromptRunning', { text: textPrompt }))
            const res = await fetch(`/api/segmentation/text_prompt`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    state_id: stateId, frame_idx: currentFrameIdx,
                    text: textPrompt.trim(), obj_id: currentObjIdRef.current
                })
            })
            if (!res.ok) throw new Error(tt('seg.textPromptFailed'))
            const data = await res.json()
            if (data.mask_png) setMaskOverlay(`data:image/png;base64,${data.mask_png}`)
            setHasPrompts(true)
            setStatus(tt('seg.textPromptApplied'))
        } catch (e: any) {
            setStatus(tt('seg.error', { detail: e.message }))
        } finally {
            setLoading(false)
        }
    }, [stateId, textPrompt, loading, currentFrameIdx])

    // ── Propagate handler (SSE streaming) ─────────────────
    const [propagationPct, setPropagationPct] = useState(-1)  // -1 = not propagating

    // ── Evaluate handler (DINOv2) ─────────────────
    const handleEvaluate = useCallback(async () => {
        if (!stateId || !hasPrompts || loading) return;
        try {
            setLoading(true)
            setStatus(tt('seg.evaluating'))
            const res = await fetch('/api/segmentation/evaluate_frames', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    state_id: stateId,
                    session_id: sessionId,
                    frame_idx: currentFrameIdx,
                    obj_id: currentObjIdRef.current
                })
            })
            if (!res.ok) throw new Error(tt('seg.evaluationFailed'))
            const data = await res.json()
            const valid = data.valid_frames as boolean[]
            const newSet = new Set<number>()
            valid.forEach((v, i) => { if (v) newSet.add(i) })
            setSelectedFrames(newSet)
            setStatus(tt('seg.evaluated', { n: newSet.size, total: valid.length }))
        } catch (e: any) {
            setStatus(tt('seg.error', { detail: e.message }))
        } finally {
            setLoading(false)
        }
    }, [stateId, sessionId, currentFrameIdx, hasPrompts, loading])

    // Queue the object currently being clicked and start a fresh one. Its prompts
    // stay in the SAM3 session under its obj_id; the canvas clears for the next
    // object. All queued objects propagate together on "Propagate all".
    const handleAddObject = useCallback(() => {
        if (!hasPrompts) return
        const objId = currentObjIdRef.current
        const name = labelName.trim() || `object_${objId}`
        const color = instanceColors()[(instancesRef.current.length + pendingObjects.length) % instanceColors().length]
        setPendingObjects(prev => [...prev, { objId, name, color }])
        currentObjIdRef.current = objId + 1
        // Clear the canvas for the next object (SAM3 keeps this object's prompts).
        setPromptPoints([])
        setMaskOverlay(null)
        setHasPrompts(false)
        setLabelName('')
        promptPointsMapRef.current.clear()
        maskCacheRef.current.clear()
        setStatus(tt('seg.queued', { name }))
    }, [hasPrompts, labelName, pendingObjects.length])

    const handlePropagate = useCallback(async () => {
        if (!stateId || propagatingRef.current) return
        // Gather every object to track at once: the queued ones + the in-edit one
        // (if it has prompts). Each keeps its own obj_id  its own saved label.
        const objs = [...pendingObjects]
        if (hasPrompts) {
            const editId = currentObjIdRef.current
            objs.push({ objId: editId, name: labelName.trim() || `object_${editId}`, color: '' })
        }
        if (objs.length === 0) return
        propagatingRef.current = true
        const objLabels: Record<number, string> = {}
        objs.forEach(o => { objLabels[o.objId] = o.name })
        const name = objs.length === 1 ? objs[0].name : tt('seg.nObjects', { n: objs.length })
        let receivedDone = false
        try {
            setLoading(true)
            setPropagationPct(0)
            setStatus(tt('seg.propagating', { name, n: selectedFrames.size }))

            const res = await fetch(`/api/segmentation/propagate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    state_id: stateId,
                    session_id: sessionId,
                    label_name: objs[0].name,   // fallback for any untracked obj_id
                    obj_labels: objLabels,       // {obj_id: name} — per-object labels
                    selected_frames: Array.from(selectedFrames).sort((a, b) => a - b)
                })
            })

            if (res.status === 409) {
                // Already propagating from a prior click — just wait
                setStatus(tt('seg.propagationInProgress'))
                return
            }
            if (!res.ok) throw new Error(tt('seg.propagationFailed'))
            if (!res.body) throw new Error(tt('seg.noResponse'))

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
                                setStatus(tt('seg.propagatingFrame', { name, frame: data.frame + 1, total: data.total, pct: data.pct }))
                                if (data.mask_png) {
                                    setMaskOverlay(`data:image/png;base64,${data.mask_png}`)
                                }
                            } else if (eventType === 'saving') {
                                setStatus(data.status || tt('seg.saving'))
                            } else if (eventType === 'done') {
                                receivedDone = true
                                isDirty.current = true
                                setPropagationPct(-1)
                                setHasPrompts(false)
                                setMaskOverlay(null)
                                setLabelName('')
                                setPromptPoints([])
                                setPendingObjects([])
                                promptPointsMapRef.current.clear()
                                maskCacheRef.current.clear()
                                setSelectedFrames(new Set(keyframes.map((_, i) => i)))
                                setStatus(tt('seg.segmented', { name }))

                                // Clear SAM3 tracked objects so next segment starts fresh
                                if (stateId) {
                                    try {
                                        await fetch('/api/segmentation/clear_prompts', {
                                            method: 'POST',
                                            headers: { 'Content-Type': 'application/json' },
                                            body: JSON.stringify({ state_id: stateId })
                                        })
                                    } catch { /* silent */ }
                                }

                                // Reload instances from server
                                try {
                                    const segRes = await fetch(`/api/sessions/${sessionId}/segmentation`)
                                    if (segRes.ok) {
                                        const segData = await segRes.json()
                                        setInstances((segData.instances || []).map((inst: any, i: number) => ({
                                            id: inst.id,
                                            label: inst.label || tt('seg.objectFallback', { id: inst.id }),
                                            color: inst.color || instanceColors()[i % instanceColors().length],
                                            total_points: inst.total_points || 0,
                                            excluded: inst.excluded || false,
                                        })))
                                    }
                                } catch (fetchErr) {
                                    console.error('Failed to reload instances:', fetchErr)
                                }
                            } else if (eventType === 'error') {
                                throw new Error(data.message || tt('seg.propagationFailed'))
                            }
                        }
                    }
                }
            }

            for (;;) {
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
                setPromptPoints([])
                setPendingObjects([])
                promptPointsMapRef.current.clear()
                maskCacheRef.current.clear()
                setSelectedFrames(new Set(keyframes.map((_, i) => i)))
                setStatus(tt('seg.propagationFinished', { name }))
                // Clear SAM3 tracked objects
                if (stateId) {
                    try {
                        await fetch('/api/segmentation/clear_prompts', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ state_id: stateId })
                        })
                    } catch { /* silent */ }
                }
                // Reload instances
                try {
                    const segRes = await fetch(`/api/sessions/${sessionId}/segmentation`)
                    if (segRes.ok) {
                        const segData = await segRes.json()
                        setInstances((segData.instances || []).map((inst: any, i: number) => ({
                            id: inst.id,
                            label: inst.label || tt('seg.objectFallback', { id: inst.id }),
                            color: inst.color || instanceColors()[i % instanceColors().length],
                            total_points: inst.total_points || 0,
                            excluded: inst.excluded || false,
                        })))
                    }
                } catch { /* silent */ }
            }
        } catch (e: any) {
            setStatus(tt('seg.error', { detail: e.message }))
            setPropagationPct(-1)
        } finally {
            setLoading(false)
            propagatingRef.current = false
        }
    }, [stateId, sessionId, labelName, selectedFrames, keyframes])

    // ── Auto segmentation ──────────────────────────────────
    const handleAutoSegment = useCallback(async () => {
        try {
            setLoading(true)
            setStatus(tt('seg.autoRunning'))
            const res = await fetch(`/api/segmentation/auto/${sessionId}`, { method: 'POST' })
            if (!res.ok) throw new Error(tt('seg.autoFailed'))
            const data = await res.json()
            isDirty.current = true
            setStatus(tt('seg.autoDone', { n: (data.instances || []).length }))

            // Reload instances
            const segRes = await fetch(`/api/sessions/${sessionId}/segmentation`)
            if (segRes.ok) {
                const segData = await segRes.json()
                setInstances((segData.instances || []).map((inst: any, i: number) => ({
                    id: inst.id,
                    label: inst.label || tt('seg.objectFallback', { id: inst.id }),
                    color: inst.color || instanceColors()[i % instanceColors().length],
                    total_points: inst.total_points || 0,
                    excluded: inst.excluded || false,
                })))
            }
        } catch (e: any) {
            setStatus(tt('seg.error', { detail: e.message }))
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
        const ok = await confirmDanger(tt('seg.deleteMessage', { label }), tt('instances.deleteTitle'))
        if (!ok) return
        try {
            setStatus(tt('seg.deleting', { label }))
            const res = await fetch(`/api/segmentation/delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                // label pins the exact entry: without it the backend falls
                // back to strict obj-id matching and entries whose obj_id ≠
                // instance_id (e.g. a half-deleted legacy row) never match
                body: JSON.stringify({ session_id: sessionId, instance_id: instId, label })
            })
            if (!res.ok) throw new Error(tt('seg.deleteFailed'))
            isDirty.current = true
            setInstances(prev => prev.filter(i => i.id !== instId))
            if (selectedInstance === instId) {
                setSelectedInstance(null)
                setMaskOverlay(null)
            }
            setStatus(tt('instances.deleted', { label }))
            onUpdate?.()
        } catch (e: any) {
            setStatus(tt('seg.error', { detail: e.message }))
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
    const t = useT()
    const fmt = useFmt()
    const clearAllPrompts = async () => {
        const n = pendingObjects.length + (hasPrompts ? 1 : 0)
        const ok = await confirmDanger(t('seg.clearAllMessage', { n: pendingObjects.length }), t('seg.clearAllTitle'))
        if (!ok) return
        setPromptPoints([]); setPendingObjects([]); setMaskOverlay(null); setHasPrompts(false); setLabelName('')
        promptPointsMapRef.current.clear(); maskCacheRef.current.clear()
        if (stateId) {
            try {
                setLoading(true)
                setStatus(t('seg.clearingAll', { n }))
                const res = await fetch('/api/segmentation/clear_prompts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ state_id: stateId }) })
                setStatus(res.ok ? t('seg.allCleared') : t('seg.clearBackendWarning'))
            } catch { setStatus(t('seg.errorClearing')) }
            finally { setLoading(false) }
        } else setStatus(t('seg.allCleared'))
    }
    const propagateLabel = pendingObjects.length > 0
        ? t('seg.propagateMany', { objects: pendingObjects.length + (hasPrompts ? 1 : 0), frames: selectedFrames.size })
        : t('seg.propagate', { frames: selectedFrames.size })

    return (
        <Dialog open size="full" title={t('seg.title')} subtitle={sessionId} icon={<Crosshair aria-hidden />} onClose={() => onClose(isDirty.current)} className="stac-seg">
            <div className="stac-seg__body">
                <aside className="stac-seg__sidebar">
                    <div className="stac-seg__sidebar-head">{t('seg.instances', { n: instances.length })}</div>
                    <div className="stac-seg__list">
                        {instances.length === 0 && <EmptyState compact title={t('seg.noInstances')} description={t('seg.noInstancesHint')} />}
                        {instances.map(inst => (
                            <div key={inst.id}
                                className={`stac-seg__inst ${selectedInstance === inst.id ? 'stac-seg__inst--selected' : ''} ${inst.excluded ? 'stac-seg__inst--excluded' : ''}`.trim()}
                                onClick={() => {
                                    const newSel = inst.id === selectedInstance ? null : inst.id
                                    setSelectedInstance(newSel)
                                    if (newSel !== null && currentFrame) fetchInstanceMask(newSel, currentFrame, kfIndex)
                                    else setMaskOverlay(null)
                                }}>
                                <ColorDot color={inst.color} />
                                <input className="stac-seg__name" value={inst.label} aria-label={t('instances.renameLabel')}
                                    onChange={e => { const val = e.target.value; setInstances(prev => prev.map(i => i.id === inst.id ? { ...i, label: val } : i)) }}
                                    onBlur={e => handleRename(inst.id, e.target.value)} onClick={e => e.stopPropagation()} />
                                <span className="stac-seg__pts stac-mono">{fmt.integer(inst.total_points)}</span>
                                <span className="stac-seg__actions" onClick={e => e.stopPropagation()}>
                                    <IconButton size="sm" label={t('seg.editMask')} icon={<Pencil aria-hidden />} active={selectedInstance === inst.id}
                                        onClick={() => {
                                            const newSel = inst.id === selectedInstance ? null : inst.id
                                            setSelectedInstance(newSel)
                                            if (newSel !== null && currentFrame) { fetchInstanceMask(newSel, currentFrame, kfIndex); setStatus(t('seg.editing', { label: inst.label })) }
                                            else setMaskOverlay(null)
                                        }} />
                                    <IconButton size="sm" label={inst.excluded ? t('seg.include') : t('seg.exclude')} icon={inst.excluded ? <EyeOff aria-hidden /> : <Eye aria-hidden />}
                                        active={!!inst.excluded} onClick={() => handleToggleExclude(inst.id, !inst.excluded)} />
                                    <IconButton size="sm" variant="danger" label={t('instances.delete')} icon={<Trash2 aria-hidden />} onClick={() => handleDelete(inst.id, inst.label)} />
                                </span>
                            </div>
                        ))}
                    </div>
                </aside>

                <section className="stac-seg__viewer">
                    <div className="stac-seg__status" role="status">
                        <span className={`stac-live-dot ${loading ? 'stac-live-dot--warn stac-live-dot--pulse' : 'stac-live-dot--ok'}`} aria-hidden />
                        <span className="stac-seg__status-text stac-mono">{status}</span>
                    </div>
                    {propagationPct >= 0 && <Progress value={propagationPct} tone="measure" size="md" className="stac-seg__progress" />}

                    <div className="stac-seg__toolbar">
                        <SegmentedControl<SegMode> ariaLabel={t('seg.mode')} value={mode} onChange={setMode} options={[
                            { value: 'manual', label: t('seg.modeManual'), icon: <MousePointer aria-hidden />, disabled: loading },
                            { value: 'text', label: t('seg.modeText'), icon: <Type aria-hidden />, disabled: loading },
                            { value: 'auto', label: t('seg.modeAuto'), icon: <Bot aria-hidden />, disabled: loading },
                        ]} />
                        <Sep />
                        <IconButton label={t('seg.prevFrame')} shortcut="ArrowLeft" icon={<ChevronLeft aria-hidden />} disabled={kfIndex <= 0} onClick={() => setKfIndex(i => i - 1)} />
                        <span className="stac-seg__nav stac-mono">
                            {t('seg.keyframeOf', { n: kfIndex + 1, total: keyframes.length })}
                            {currentFrame && <span className="stac-seg__filename">{currentFrame}</span>}
                        </span>
                        <IconButton label={t('seg.nextFrame')} shortcut="ArrowRight" icon={<ChevronRight aria-hidden />} disabled={kfIndex >= keyframes.length - 1} onClick={() => setKfIndex(i => i + 1)} />
                        <Sep />
                        <IconButton label={t('seg.zoomIn')} icon={<ZoomIn aria-hidden />} onClick={() => setZoom(z => Math.min(8, z + 0.5))} />
                        <span className="stac-seg__zoom stac-mono">{fmt.percent(zoom)}</span>
                        <IconButton label={t('seg.zoomOut')} icon={<ZoomOut aria-hidden />} onClick={() => setZoom(z => Math.max(0.5, z - 0.5))} />
                        <IconButton label={t('seg.resetView')} icon={<Maximize aria-hidden />} onClick={resetView} />
                        <Sep />
                        <IconButton label={t('seg.includeFrameNext')} icon={<Check aria-hidden />} tone="brand" onClick={() => {
                            const next = new Set(selectedFrames); next.add(kfIndex); setSelectedFrames(next)
                            if (kfIndex < keyframes.length - 1) setKfIndex(i => i + 1)
                        }} />
                        <IconButton label={t('seg.excludeFrameNext')} icon={<X aria-hidden />} variant="danger" onClick={() => {
                            const next = new Set(selectedFrames); next.delete(kfIndex); setSelectedFrames(next)
                            if (kfIndex < keyframes.length - 1) setKfIndex(i => i + 1)
                        }} />
                    </div>

                    {mode === 'text' && (
                        <div className="stac-seg__bar">
                            <Input placeholder={t('seg.textPlaceholder')} value={textPrompt} onChange={e => setTextPrompt(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleTextPrompt()} disabled={loading} />
                            <Button variant="primary" onClick={handleTextPrompt} disabled={loading || !textPrompt.trim()}>{t('seg.segment')}</Button>
                        </div>
                    )}
                    {mode === 'auto' && (
                        <div className="stac-seg__bar">
                            <span className="stac-seg__hint">{t('seg.autoDesc')}</span>
                            <Button variant="primary" icon={<Bot aria-hidden />} onClick={handleAutoSegment} loading={loading}>{t('seg.startAuto')}</Button>
                        </div>
                    )}

                    <div ref={canvasRef} className={`stac-seg__canvas ${loading ? 'stac-seg__canvas--loading' : ''} ${mode === 'manual' ? 'stac-seg__canvas--crosshair' : ''}`.trim()}
                        onWheel={handleWheel} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
                        {currentFrame ? (
                            <div className="stac-seg__transform" style={{ '--pan-x': `${pan.x}px`, '--pan-y': `${pan.y}px`, '--zoom': zoom } as React.CSSProperties}>
                                <div className="stac-seg__image-wrap">
                                    <img ref={imgRef} src={`/api/sessions/${sessionId}/frames/${currentFrame}`} className="stac-seg__image" alt={t('seg.keyframeAlt', { name: currentFrame })} draggable={false} onLoad={() => redrawCanvas()} />
                                    <canvas ref={overlayCanvasRef} className="stac-seg__overlay" onClick={e => handleImageClick(e, true)} onContextMenu={e => handleImageClick(e, false)} onMouseMove={handleCanvasMouseMove} />
                                </div>
                            </div>
                        ) : <EmptyState title={t('seg.noKeyframes')} />}
                    </div>

                    {mode === 'manual' && pendingObjects.length > 0 && (
                        <div className="stac-seg__queue">
                            <span className="stac-seg__hint">{t('seg.queuedCount', { n: pendingObjects.length })}</span>
                            {pendingObjects.map(o => <Badge key={o.objId} color={o.color}>{o.name}</Badge>)}
                            <span className="stac-seg__hint">{t('seg.queueHint')}</span>
                        </div>
                    )}

                    <div className="stac-seg__footer">
                        <span className="stac-seg__hint stac-seg__hint--grow">
                            {mode === 'manual' && t('seg.hintManual')}
                            {mode === 'text' && t('seg.hintText')}
                            {mode === 'auto' && t('seg.hintAuto')}
                        </span>
                        <Input size="sm" className="stac-seg__label" placeholder={t('seg.labelPlaceholder')} value={labelName} onChange={e => setLabelName(e.target.value)} disabled={loading} />
                        <Tooltip label={t('seg.clearHint')}><span><Button size="sm" icon={<X aria-hidden />} onClick={clearCurrentObject} disabled={loading || !hasPrompts}>{t('common.clear')}</Button></span></Tooltip>
                        <Tooltip label={t('seg.clearAllHint')}><span><Button size="sm" variant="danger" icon={<Trash2 aria-hidden />} onClick={clearAllPrompts} disabled={loading || (!hasPrompts && pendingObjects.length === 0)}>{t('seg.clearAll')}</Button></span></Tooltip>
                        <Button size="sm" icon={<Brain aria-hidden />} onClick={handleEvaluate} disabled={loading || !stateId || !hasPrompts || propagatingRef.current}>{t('seg.evaluate')}</Button>
                        <Tooltip label={t('seg.addObjectHint')}><span><Button size="sm" icon={<Plus aria-hidden />} onClick={handleAddObject} disabled={loading || !stateId || !hasPrompts}>{t('seg.addObject')}</Button></span></Tooltip>
                        <Button size="sm" variant="primary" icon={<Play aria-hidden />} onClick={handlePropagate} disabled={loading || !stateId || (!hasPrompts && pendingObjects.length === 0)}>{propagateLabel}</Button>
                    </div>

                    <div className="stac-seg__frames">
                        {keyframes.map((kf, i) => (
                            <div key={i} ref={el => { thumbRefs.current[i] = el }}
                                className={`stac-seg__thumb ${kfIndex === i ? 'stac-seg__thumb--current' : ''} ${selectedFrames.has(i) ? '' : 'stac-seg__thumb--off'}`.trim()}
                                onClick={() => setKfIndex(i)}>
                                <img className="stac-seg__thumb-img" src={`/api/sessions/${sessionId}/frames/${kf}`} loading="lazy" alt={t('seg.keyframeAlt', { name: kf })} />
                                {[...promptPointsMapRef.current.keys()].some(k => k.startsWith(`${i}:`)) && <span className="stac-seg__thumb-dot stac-live-dot stac-live-dot--ok" title={t('seg.frameHasPrompts')} />}
                                <span className="stac-seg__thumb-check" onClick={e => e.stopPropagation()}>
                                    <Checkbox checked={selectedFrames.has(i)} aria-label={t('seg.includeFrame', { n: i + 1 })}
                                        onChange={() => setSelectedFrames(prev => { const next = new Set(prev); if (next.has(i)) next.delete(i); else next.add(i); return next })} />
                                </span>
                                <span className="stac-seg__thumb-idx stac-mono">{i + 1}</span>
                            </div>
                        ))}
                    </div>
                </section>
            </div>
            {dialogElement}
        </Dialog>
    )
}
