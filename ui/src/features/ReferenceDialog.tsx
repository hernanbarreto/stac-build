/**
 * ReferenceDialog — hand-measured REFERENCE DISTANCES and REFERENCE PLANES
 * for the evaluation battery, anchored on KEYFRAME IMAGES (never on the
 * cloud): the user clicks the two ends of a tape / laser / total-station
 * distance on a keyframe (and may add more observations of the same ends on
 * other keyframes), or draws a polygon over a plane. The document is read
 * and written through `/api/evaluation/references/{session}`; the server
 * validates it and answers with problems (400) or counts + warnings.
 *
 * Image mechanics (zoom, pan, click to native pixel) follow
 * InteractiveSegmentation; the image is fitted to the container in JS so the
 * overlay canvas always covers it exactly, at any zoom.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Maximize, Pentagon, Pin, Plus, Save, Trash2, Undo2, X, ZoomIn, ZoomOut } from 'lucide-react'
import { ConfirmDialog, Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { IconButton } from '../components/ui/IconButton'
import { Badge } from '../components/ui/Badge'
import { Banner } from '../components/ui/Banner'
import { EmptyState } from '../components/ui/EmptyState'
import { Field, Input, NumberInput, SegmentedControl, Select } from '../components/ui/Field'
import { Row, Stack } from '../components/ui/Panel'
import { useToast } from '../components/ui/Toast'
import { tokenColor, VP } from '../components/viewport/palette'
import { useFmt, useT } from '../i18n'
import {
  emptyReferenceDocument, nextReferenceId, normalizeReferenceDocument, referenceCounts,
  type RefDistance, type RefObservation, type RefPlane, type RefPolygon, type RefRole, type RefTool, type ReferenceDocument,
} from './referenceTypes'

export type ReferenceMode = 'distance' | 'plane'

interface ReferenceDialogProps {
  open: boolean
  sessionId: string
  /** the active scan ("date/source") or null for the session's default */
  scanKey: string | null
  /** the mode the opener asked for; the user can switch inside */
  initialMode: ReferenceMode
  onClose: () => void
}

const ZOOM_MIN = 0.5, ZOOM_MAX = 8, ZOOM_STEP = 0.2
const DRAG_PX = 5
const DOUBLE_CLICK_MS = 250
const DEFAULT_PLANE_LABEL = 'floor'

const isTyping = (el: EventTarget | null) => {
  const n = el as HTMLElement | null
  return !!n && (n.tagName === 'INPUT' || n.tagName === 'TEXTAREA' || n.tagName === 'SELECT' || n.isContentEditable)
}

export function ReferenceDialog({ open, sessionId, scanKey, initialMode, onClose }: ReferenceDialogProps) {
  const t = useT()
  const fmt = useFmt()
  const { notify } = useToast()

  // ── document + keyframes ──
  const [keyframes, setKeyframes] = useState<string[]>([])
  const [kfIndex, setKfIndex] = useState(0)
  const [doc, setDoc] = useState<ReferenceDocument>(emptyReferenceDocument)
  const [path, setPath] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [warnings, setWarnings] = useState<string[]>([])
  const [problems, setProblems] = useState<string[]>([])
  const [discardOpen, setDiscardOpen] = useState(false)

  // ── editing state ──
  const [mode, setMode] = useState<ReferenceMode>(initialMode)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  /** distance / plane that receives the next observation pair / polygon (null = a new one is created) */
  const [armedId, setArmedId] = useState<string | null>(null)
  const [pendingA, setPendingA] = useState<RefObservation | null>(null)
  const [polygon, setPolygon] = useState<[number, number][]>([])
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)

  // ── zoom / pan ──
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isPanning, setIsPanning] = useState(false)
  const panStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 })
  const mouseDownPos = useRef<{ x: number; y: number } | null>(null)
  const wasDragging = useRef(false)
  const lastClickTime = useRef(0)

  const imgRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const thumbRefs = useRef<(HTMLButtonElement | null)[]>([])

  const currentFile = keyframes[kfIndex] ?? null
  const currentFrame = currentFile ? parseInt(currentFile, 10) : null
  const counts = useMemo(() => referenceCounts(doc), [doc])
  const selectedDistance = useMemo(() => doc.distances.find(d => d.id === selectedId) ?? null, [doc, selectedId])
  const selectedPlane = useMemo(() => doc.planes.find(p => p.id === selectedId) ?? null, [doc, selectedId])

  /** frames that hold at least one observation or polygon (thumbnail dots) */
  const framesWithData = useMemo(() => {
    const s = new Set<number>()
    for (const d of doc.distances) for (const end of d.ends) for (const o of end.observations) s.add(o.frame)
    for (const p of doc.planes) for (const g of p.polygons) s.add(g.frame)
    return s
  }, [doc])

  const update = useCallback((fn: (d: ReferenceDocument) => ReferenceDocument) => { setDoc(fn); setDirty(true) }, [])

  // ── load on open ──
  useEffect(() => {
    if (!open) return
    let cancelled = false
    setMode(initialMode); setSelectedId(null); setArmedId(null); setPendingA(null); setPolygon([]); setConfirmDeleteId(null)
    setDirty(false); setWarnings([]); setProblems([]); setLoadError(null); setDiscardOpen(false)
    setZoom(1); setPan({ x: 0, y: 0 }); setKfIndex(0); setKeyframes([]); setDoc(emptyReferenceDocument()); setPath(null)
    setLoading(true)
    const q = scanKey ? `?scan_key=${encodeURIComponent(scanKey)}` : ''
    const getJson = async (url: string) => {
      const r = await fetch(url)
      if (!r.ok) throw new Error(`${r.status} ${url}`)
      return r.json()
    }
    Promise.all([getJson(`/api/sessions/${sessionId}/keyframes`), getJson(`/api/evaluation/references/${sessionId}${q}`)])
      .then(([kf, ref]) => {
        if (cancelled) return
        setKeyframes(Array.isArray(kf?.keyframes) ? kf.keyframes.filter((n: unknown) => typeof n === 'string') : [])
        setDoc(normalizeReferenceDocument(ref?.document))
        setPath(typeof ref?.path === 'string' ? ref.path : null)
      })
      .catch((e: any) => { if (!cancelled) setLoadError(e?.message ?? String(e)) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [open, sessionId, scanKey, initialMode])

  // a polygon in progress belongs to the keyframe it was started on
  useEffect(() => { setPolygon([]) }, [kfIndex])
  // switching mode drops what was half done
  useEffect(() => { setPendingA(null); setPolygon([]); setArmedId(null); setConfirmDeleteId(null) }, [mode])

  useEffect(() => {
    thumbRefs.current[kfIndex]?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' })
  }, [kfIndex])

  // ── overlay drawing ──
  const redraw = useCallback(() => {
    const canvas = overlayRef.current, img = imgRef.current
    if (!canvas || !img || !img.naturalWidth) return
    const w = img.clientWidth, h = img.clientHeight
    if (w <= 0 || h <= 0) return
    const dpr = window.devicePixelRatio || 1
    const k = Math.min(dpr * zoom, 4)
    const bw = Math.round(w * k), bh = Math.round(h * k)
    if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh }
    canvas.style.width = `${w}px`
    canvas.style.height = `${h}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(k, 0, 0, k, 0, 0)
    ctx.clearRect(0, 0, w, h)
    const sx = w / img.naturalWidth, sy = h / img.naturalHeight
    const px = 1 / zoom // layout px per screen px: markers keep their size on screen
    const fontPx = (parseFloat(tokenColor('--fs-1')) || 12) * px
    const font = `${fontPx}px ${tokenColor('--font-sans')}`
    const colA = tokenColor(VP.ok), colB = tokenColor(VP.measure), colPlane = tokenColor(VP.warn), colNew = tokenColor(VP.brand)
    const colText = tokenColor(VP.text1), colBg = tokenColor(VP.labelBg), colRing = tokenColor(VP.light)
    const hasSelection = selectedId != null

    const label = (text: string, x: number, y: number) => {
      ctx.font = font
      const m = ctx.measureText(text)
      const padX = 4 * px, lh = fontPx * 1.4
      ctx.fillStyle = colBg
      ctx.fillRect(x, y - lh, m.width + padX * 2, lh)
      ctx.fillStyle = colText
      ctx.textBaseline = 'middle'
      ctx.fillText(text, x + padX, y - lh / 2)
    }
    const marker = (x: number, y: number, color: string, r: number, dashed = false) => {
      ctx.beginPath()
      ctx.arc(x, y, r, 0, Math.PI * 2)
      if (dashed) { ctx.setLineDash([3 * px, 3 * px]); ctx.strokeStyle = color; ctx.lineWidth = 2 * px; ctx.stroke(); ctx.setLineDash([]) }
      else { ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = colRing; ctx.lineWidth = 1.5 * px; ctx.stroke() }
    }

    // distances: observations of each end on this keyframe, a line when both ends are here
    for (const d of doc.distances) {
      const onFrame = (i: 0 | 1) => d.ends[i].observations.filter(o => o.frame === currentFrame)
      const a = onFrame(0), b = onFrame(1)
      if (!a.length && !b.length) continue
      const selected = d.id === selectedId
      ctx.globalAlpha = hasSelection && !selected ? 0.55 : 1
      if (a.length && b.length) {
        ctx.beginPath()
        ctx.moveTo(a[0].u * sx, a[0].v * sy)
        ctx.lineTo(b[0].u * sx, b[0].v * sy)
        ctx.strokeStyle = selected ? colNew : colB
        ctx.lineWidth = (selected ? 2.5 : 1.5) * px
        ctx.stroke()
      }
      for (const o of a) marker(o.u * sx, o.v * sy, colA, (selected ? 6 : 5) * px)
      for (const o of b) marker(o.u * sx, o.v * sy, colB, (selected ? 6 : 5) * px)
      const anchor = a[0] ?? b[0]
      label(`${d.id}  ${d.value_m > 0 ? fmt.lengthText(d.value_m) : t('status.dash')}`, anchor.u * sx + 8 * px, anchor.v * sy - 8 * px)
      ctx.globalAlpha = 1
    }
    if (pendingA && pendingA.frame === currentFrame) marker(pendingA.u * sx, pendingA.v * sy, colA, 7 * px, true)

    // planes: closed, lightly filled polygons on this keyframe
    for (const p of doc.planes) {
      const selected = p.id === selectedId
      ctx.globalAlpha = hasSelection && !selected ? 0.55 : 1
      for (const g of p.polygons) {
        if (g.frame !== currentFrame || g.points.length < 3) continue
        ctx.beginPath()
        g.points.forEach(([u, v], i) => { if (i === 0) ctx.moveTo(u * sx, v * sy); else ctx.lineTo(u * sx, v * sy) })
        ctx.closePath()
        ctx.globalAlpha = (hasSelection && !selected ? 0.55 : 1) * 0.18
        ctx.fillStyle = colPlane
        ctx.fill()
        ctx.globalAlpha = hasSelection && !selected ? 0.55 : 1
        ctx.strokeStyle = selected ? colNew : colPlane
        ctx.lineWidth = (selected ? 2.5 : 1.5) * px
        ctx.stroke()
        for (const [u, v] of g.points) marker(u * sx, v * sy, colPlane, 3.5 * px)
        const cx = g.points.reduce((s, q) => s + q[0], 0) / g.points.length * sx
        const cy = g.points.reduce((s, q) => s + q[1], 0) / g.points.length * sy
        label(`${p.id}  ${p.label}`, cx, cy)
      }
      ctx.globalAlpha = 1
    }

    // the polygon being drawn
    if (polygon.length) {
      ctx.beginPath()
      polygon.forEach(([u, v], i) => { if (i === 0) ctx.moveTo(u * sx, v * sy); else ctx.lineTo(u * sx, v * sy) })
      ctx.setLineDash([4 * px, 3 * px])
      ctx.strokeStyle = colNew
      ctx.lineWidth = 2 * px
      ctx.stroke()
      ctx.setLineDash([])
      for (const [u, v] of polygon) marker(u * sx, v * sy, colNew, 4 * px)
    }
  }, [doc, selectedId, pendingA, polygon, currentFrame, zoom, fmt, t])

  /** size the image to fit the container (its layout size, so the overlay covers it at any zoom) */
  const fitImage = useCallback(() => {
    const img = imgRef.current, box = containerRef.current
    if (!img || !box || !img.naturalWidth) return
    const cw = box.clientWidth, ch = box.clientHeight
    if (cw <= 0 || ch <= 0) return
    const s = Math.min(cw / img.naturalWidth, ch / img.naturalHeight)
    img.style.width = `${Math.max(1, Math.floor(img.naturalWidth * s))}px`
    img.style.height = `${Math.max(1, Math.floor(img.naturalHeight * s))}px`
    redraw()
  }, [redraw])
  const fitRef = useRef(fitImage)
  fitRef.current = fitImage

  useEffect(() => { redraw() }, [redraw])

  useEffect(() => {
    if (!open) return
    const box = containerRef.current
    if (!box || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => fitRef.current())
    ro.observe(box)
    return () => ro.disconnect()
  }, [open, loading])

  const onImageLoad = () => {
    const img = imgRef.current
    if (img && img.naturalWidth > 0) {
      const size: [number, number] = [img.naturalWidth, img.naturalHeight]
      setDoc(d => (d.image_size ? d : { ...d, image_size: size }))
    }
    fitImage()
  }

  // ── placing ──
  const cancelPending = useCallback(() => { setPendingA(null); setPolygon([]); setArmedId(null); setConfirmDeleteId(null) }, [])

  const placeDistancePoint = useCallback((obs: RefObservation) => {
    if (!pendingA) { setPendingA(obs); return }
    const a = pendingA, b = obs
    if (armedId && doc.distances.some(d => d.id === armedId)) {
      update(d => ({ ...d, distances: d.distances.map(x => x.id === armedId
        ? { ...x, ends: [{ observations: [...x.ends[0].observations, a] }, { observations: [...x.ends[1].observations, b] }] }
        : x) }))
      setSelectedId(armedId)
    } else {
      const id = nextReferenceId('d', doc.distances)
      const fresh: RefDistance = { id, value_m: 0, sigma_m: 0, role: 'test', tool: 'tape', note: '', ends: [{ observations: [a] }, { observations: [b] }] }
      update(d => ({ ...d, distances: [...d.distances, fresh] }))
      setSelectedId(id)
    }
    setPendingA(null)
    setArmedId(null)
  }, [pendingA, armedId, doc.distances, update])

  const closePolygon = useCallback(() => {
    if (polygon.length < 3 || currentFrame == null || !currentFile) return
    // a double-click on the last vertex adds it twice: drop a last point that sits on the previous one
    const natW = imgRef.current?.naturalWidth || 1000
    const [lu, lv] = polygon[polygon.length - 1], [pu, pv] = polygon[polygon.length - 2]
    const dup = polygon.length > 3 && Math.hypot(lu - pu, lv - pv) < natW * 0.003
    const points = dup ? polygon.slice(0, -1) : polygon
    const poly: RefPolygon = { frame: currentFrame, file: currentFile, points }
    if (armedId && doc.planes.some(p => p.id === armedId)) {
      update(d => ({ ...d, planes: d.planes.map(p => (p.id === armedId ? { ...p, polygons: [...p.polygons, poly] } : p)) }))
      setSelectedId(armedId)
    } else {
      const id = nextReferenceId('p', doc.planes)
      const fresh: RefPlane = { id, label: DEFAULT_PLANE_LABEL, note: '', polygons: [poly] }
      update(d => ({ ...d, planes: [...d.planes, fresh] }))
      setSelectedId(id)
    }
    setPolygon([])
    setArmedId(null)
  }, [polygon, currentFrame, currentFile, armedId, doc.planes, update])

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    e.stopPropagation()
    if (wasDragging.current) { wasDragging.current = false; return }
    const now = Date.now()
    if (now - lastClickTime.current < DOUBLE_CLICK_MS) return // the second click of a double-click
    lastClickTime.current = now
    const img = imgRef.current, canvas = overlayRef.current
    if (!img || !canvas || currentFrame == null || !currentFile || loading) return
    const rect = canvas.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return
    const u = (e.clientX - rect.left) * img.naturalWidth / rect.width
    const v = (e.clientY - rect.top) * img.naturalHeight / rect.height
    if (mode === 'distance') placeDistancePoint({ frame: currentFrame, file: currentFile, u, v })
    else setPolygon(p => [...p, [u, v]])
  }, [currentFrame, currentFile, loading, mode, placeDistancePoint])

  const handleCanvasDoubleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    if (mode === 'plane') closePolygon()
  }, [mode, closePolygon])

  // ── keyboard: arrows change the keyframe, Enter closes the polygon, Backspace drops its last
  // vertex, Escape cancels what is half done (captured before the Dialog's own Escape) ──
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target) || discardOpen) return
      if (e.key === 'ArrowLeft') { e.preventDefault(); setKfIndex(i => Math.max(0, i - 1)) }
      else if (e.key === 'ArrowRight') { e.preventDefault(); setKfIndex(i => Math.min(Math.max(0, keyframes.length - 1), i + 1)) }
      else if (e.key === 'Enter') { if (mode === 'plane' && polygon.length >= 3) { e.preventDefault(); closePolygon() } }
      else if (e.key === 'Backspace') { if (mode === 'plane' && polygon.length) { e.preventDefault(); setPolygon(p => p.slice(0, -1)) } }
      else if (e.key === 'Escape') {
        if (pendingA || polygon.length || armedId || confirmDeleteId) { e.preventDefault(); e.stopPropagation(); cancelPending() }
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [open, discardOpen, keyframes.length, mode, polygon, pendingA, armedId, confirmDeleteId, closePolygon, cancelPending])

  // ── zoom / pan (wheel registered natively: the dialog lives in a body portal, where
  // React's wheel listener is passive and preventDefault is refused) ──
  useEffect(() => {
    const box = containerRef.current
    if (!box || !open) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      setZoom(z => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, z + (e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP))))
    }
    box.addEventListener('wheel', onWheel, { passive: false })
    return () => box.removeEventListener('wheel', onWheel)
  }, [open, loading])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 1 || (e.shiftKey && e.button === 0)) {
      e.preventDefault()
      setIsPanning(true)
      panStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y }
    }
    mouseDownPos.current = { x: e.clientX, y: e.clientY }
  }, [pan])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning) return
    setPan({ x: panStart.current.panX + (e.clientX - panStart.current.x), y: panStart.current.panY + (e.clientY - panStart.current.y) })
  }, [isPanning])

  const handleMouseUp = useCallback((e: React.MouseEvent) => {
    if (mouseDownPos.current) {
      const dx = e.clientX - mouseDownPos.current.x, dy = e.clientY - mouseDownPos.current.y
      wasDragging.current = Math.sqrt(dx * dx + dy * dy) > DRAG_PX
    }
    setIsPanning(false)
  }, [])

  const resetView = useCallback(() => { setZoom(1); setPan({ x: 0, y: 0 }) }, [])

  // ── list / form actions ──
  const goToItem = useCallback((frames: number[]) => {
    if (currentFrame != null && frames.includes(currentFrame)) return
    const idx = keyframes.findIndex(n => frames.includes(parseInt(n, 10)))
    if (idx >= 0) setKfIndex(idx)
  }, [currentFrame, keyframes])

  const selectDistance = (d: RefDistance) => {
    setSelectedId(d.id); setArmedId(null); setPendingA(null); setConfirmDeleteId(null)
    if (mode !== 'distance') setMode('distance')
    goToItem(d.ends.flatMap(e => e.observations.map(o => o.frame)))
  }
  const selectPlane = (p: RefPlane) => {
    setSelectedId(p.id); setArmedId(null); setPolygon([]); setConfirmDeleteId(null)
    if (mode !== 'plane') setMode('plane')
    goToItem(p.polygons.map(g => g.frame))
  }
  const patchDistance = (id: string, patch: Partial<RefDistance>) => update(d => ({ ...d, distances: d.distances.map(x => (x.id === id ? { ...x, ...patch } : x)) }))
  const patchPlane = (id: string, patch: Partial<RefPlane>) => update(d => ({ ...d, planes: d.planes.map(x => (x.id === id ? { ...x, ...patch } : x)) }))
  const removeObservation = (id: string, end: 0 | 1, index: number) => update(d => ({ ...d, distances: d.distances.map(x => {
    if (x.id !== id) return x
    const ends: [typeof x.ends[0], typeof x.ends[1]] = [{ observations: [...x.ends[0].observations] }, { observations: [...x.ends[1].observations] }]
    ends[end].observations.splice(index, 1)
    return { ...x, ends }
  }) }))
  const removePolygon = (id: string, index: number) => update(d => ({ ...d, planes: d.planes.map(x => (x.id === id ? { ...x, polygons: x.polygons.filter((_, i) => i !== index) } : x)) }))
  const deleteItem = (id: string) => {
    update(d => ({ ...d, distances: d.distances.filter(x => x.id !== id), planes: d.planes.filter(x => x.id !== id) }))
    if (selectedId === id) setSelectedId(null)
    if (armedId === id) setArmedId(null)
    setConfirmDeleteId(null)
  }

  // ── save ──
  const save = async () => {
    setSaving(true)
    setProblems([])
    try {
      const r = await fetch(`/api/evaluation/references/${sessionId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scan_key: scanKey, document: doc }),
      })
      const data = await r.json().catch(() => null)
      if (r.status === 400) {
        const det = data?.detail
        const list: string[] = Array.isArray(det?.problems) ? det.problems.map(String) : []
        setProblems(list)
        notify(t('reference.invalid', { error: typeof det?.error === 'string' ? det.error : t('reference.invalidGeneric'), n: list.length }), { tone: 'err' })
        return
      }
      if (!r.ok) throw new Error(`${r.status}`)
      setDirty(false)
      if (typeof data?.path === 'string') setPath(data.path)
      setWarnings(Array.isArray(data?.warnings) ? data.warnings.map(String) : [])
      const c = data?.counts ?? counts
      notify(t('reference.saved', { distances: c.distances ?? 0, gauge: c.gauge ?? 0, test: c.test ?? 0, planes: c.planes ?? 0 }), { tone: 'ok' })
    } catch (e: any) {
      notify(t('reference.saveFailed', { detail: e?.message ?? String(e) }), { tone: 'err' })
    } finally {
      setSaving(false)
    }
  }

  const requestClose = () => {
    if (saving) return
    if (dirty) { setDiscardOpen(true); return }
    onClose()
  }

  // ── derived text ──
  const hint = (() => {
    if (loading) return t('common.loading')
    if (mode === 'distance') {
      if (armedId) return pendingA ? t('reference.hintObservationB', { id: armedId, frame: pendingA.frame }) : t('reference.hintObservationA', { id: armedId })
      return pendingA ? t('reference.hintDistanceB', { frame: pendingA.frame }) : t('reference.hintDistanceA')
    }
    return armedId ? t('reference.hintPlaneFor', { id: armedId, n: polygon.length }) : t('reference.hintPlane', { n: polygon.length })
  })()
  const roleLabel = (r: RefRole) => (r === 'gauge' ? t('reference.roleGauge') : t('reference.roleTest'))
  const toolLabel = (x: RefTool) => (x === 'tape' ? t('reference.toolTape') : x === 'laser' ? t('reference.toolLaser') : t('reference.toolTotalStation'))
  const pendingSomething = !!pendingA || polygon.length > 0 || !!armedId

  const renderDistanceForm = (d: RefDistance) => (
    <div className="stac-ref__form">
      <Row justify="between">
        <span className="stac-mono">{d.id}</span>
        <Badge size="sm" tone={d.role === 'gauge' ? 'brand' : 'measure'}>{d.role}</Badge>
      </Row>
      <Field label={t('reference.value')} inline>
        <NumberInput size="sm" unit="m" min={0} step={0.001} value={d.value_m} onChange={v => patchDistance(d.id, { value_m: v === '' ? 0 : v })} />
      </Field>
      <Field label={t('reference.sigma')} inline>
        <NumberInput size="sm" unit="m" min={0} step={0.001} value={d.sigma_m} onChange={v => patchDistance(d.id, { sigma_m: v === '' ? 0 : v })} />
      </Field>
      <Field label={t('reference.role')} hint={t('reference.roleHint')}>
        <SegmentedControl<RefRole> size="sm" ariaLabel={t('reference.role')} value={d.role} onChange={role => patchDistance(d.id, { role })}
          options={[{ value: 'gauge', label: roleLabel('gauge') }, { value: 'test', label: roleLabel('test') }]} />
      </Field>
      <Field label={t('reference.tool')} inline>
        <Select<RefTool> size="sm" value={d.tool} onChange={tool => patchDistance(d.id, { tool })}
          options={[{ value: 'tape', label: toolLabel('tape') }, { value: 'laser', label: toolLabel('laser') }, { value: 'total_station', label: toolLabel('total_station') }]} />
      </Field>
      <Field label={t('reference.note')}>
        <Input size="sm" value={d.note} onChange={e => patchDistance(d.id, { note: e.target.value })} />
      </Field>
      {([0, 1] as const).map(end => (
        <div key={end} className="stac-ref__chips">
          <span className="stac-ref__chips-label">{end === 0 ? t('reference.endA') : t('reference.endB')}</span>
          {d.ends[end].observations.length === 0 && <span className="stac-muted">{t('status.dash')}</span>}
          {d.ends[end].observations.map((o, i) => (
            <Badge key={`${o.frame}-${i}`} size="sm" mono tone={end === 0 ? 'ok' : 'measure'}>
              <button type="button" className="stac-ref__chip-go" onClick={() => goToItem([o.frame])} title={t('reference.goToKeyframe', { frame: o.frame })}>{o.frame}</button>
              <button type="button" className="stac-ref__chip-x" aria-label={t('reference.removeObservation', { frame: o.frame })} onClick={() => removeObservation(d.id, end, i)}><X aria-hidden /></button>
            </Badge>
          ))}
        </div>
      ))}
      <Row wrap>
        <Button size="sm" icon={<Plus aria-hidden />} active={armedId === d.id} onClick={() => { setPendingA(null); setArmedId(armedId === d.id ? null : d.id) }}>
          {armedId === d.id ? t('reference.cancelArm') : t('reference.addObservation')}
        </Button>
        {confirmDeleteId === d.id ? (
          <>
            <Button size="sm" variant="danger" icon={<Trash2 aria-hidden />} onClick={() => deleteItem(d.id)}>{t('reference.confirmDelete')}</Button>
            <Button size="sm" variant="ghost" onClick={() => setConfirmDeleteId(null)}>{t('common.cancel')}</Button>
          </>
        ) : (
          <Button size="sm" variant="ghost" icon={<Trash2 aria-hidden />} onClick={() => setConfirmDeleteId(d.id)}>{t('common.delete')}</Button>
        )}
      </Row>
    </div>
  )

  const renderPlaneForm = (p: RefPlane) => (
    <div className="stac-ref__form">
      <Row justify="between">
        <span className="stac-mono">{p.id}</span>
        <Badge size="sm" tone="warn">{t('reference.polygons', { n: p.polygons.length })}</Badge>
      </Row>
      <Field label={t('reference.label')} inline>
        <Input size="sm" value={p.label} onChange={e => patchPlane(p.id, { label: e.target.value })} />
      </Field>
      <Field label={t('reference.note')}>
        <Input size="sm" value={p.note} onChange={e => patchPlane(p.id, { note: e.target.value })} />
      </Field>
      <div className="stac-ref__chips">
        <span className="stac-ref__chips-label">{t('reference.polygonsLabel')}</span>
        {p.polygons.length === 0 && <span className="stac-muted">{t('status.dash')}</span>}
        {p.polygons.map((g, i) => (
          <Badge key={`${g.frame}-${i}`} size="sm" mono tone="warn">
            <button type="button" className="stac-ref__chip-go" onClick={() => goToItem([g.frame])} title={t('reference.goToKeyframe', { frame: g.frame })}>{t('reference.polygonChip', { frame: g.frame, n: g.points.length })}</button>
            <button type="button" className="stac-ref__chip-x" aria-label={t('reference.removePolygon', { frame: g.frame })} onClick={() => removePolygon(p.id, i)}><X aria-hidden /></button>
          </Badge>
        ))}
      </div>
      <Row wrap>
        <Button size="sm" icon={<Plus aria-hidden />} active={armedId === p.id} onClick={() => { setPolygon([]); setArmedId(armedId === p.id ? null : p.id) }}>
          {armedId === p.id ? t('reference.cancelArm') : t('reference.addPolygon')}
        </Button>
        {confirmDeleteId === p.id ? (
          <>
            <Button size="sm" variant="danger" icon={<Trash2 aria-hidden />} onClick={() => deleteItem(p.id)}>{t('reference.confirmDelete')}</Button>
            <Button size="sm" variant="ghost" onClick={() => setConfirmDeleteId(null)}>{t('common.cancel')}</Button>
          </>
        ) : (
          <Button size="sm" variant="ghost" icon={<Trash2 aria-hidden />} onClick={() => setConfirmDeleteId(p.id)}>{t('common.delete')}</Button>
        )}
      </Row>
    </div>
  )

  return (
    <>
      <Dialog open={open} size="full" className="stac-ref" title={t('reference.title')} subtitle={scanKey ? `${sessionId} ${scanKey}` : sessionId}
        icon={mode === 'distance' ? <Pin aria-hidden /> : <Pentagon aria-hidden />} onClose={requestClose} busy={saving}
        footer={
          <>
            <span className="stac-ref__footer-path">{path ?? ''}</span>
            {dirty && <Badge size="sm" tone="warn">{t('reference.unsaved')}</Badge>}
            <Button onClick={requestClose} disabled={saving}>{t('common.close')}</Button>
            <Button variant="primary" icon={<Save aria-hidden />} loading={saving} disabled={!dirty || loading || !!loadError} onClick={save}>{t('common.save')}</Button>
          </>
        }>
        <div className="stac-ref__body">
          <aside className="stac-ref__side">
            <div className="stac-ref__side-head">
              <SegmentedControl<ReferenceMode> size="sm" ariaLabel={t('reference.mode')} value={mode} onChange={setMode} options={[
                { value: 'distance', icon: <Pin aria-hidden />, label: t('reference.modeDistance') },
                { value: 'plane', icon: <Pentagon aria-hidden />, label: t('reference.modePlane') },
              ]} />
              <div className="stac-ref__counts">{t('reference.counts', counts)}</div>
              {!loading && !loadError && counts.gauge === 0 && <Banner tone="warn" compact>{t('reference.noGauge')}</Banner>}
              {!loading && !loadError && counts.test === 0 && <Banner tone="warn" compact>{t('reference.noTest')}</Banner>}
              {loadError && <Banner tone="err" compact>{t('reference.loadFailed', { detail: loadError })}</Banner>}
              {problems.length > 0 && (
                <Banner tone="err" compact title={t('reference.problems', { n: problems.length })}>
                  <Stack gap={1}>{problems.map((p, i) => <span key={i}>{p}</span>)}</Stack>
                </Banner>
              )}
              {warnings.length > 0 && (
                <Banner tone="warn" compact title={t('reference.warnings', { n: warnings.length })} onClose={() => setWarnings([])}>
                  <Stack gap={1}>{warnings.map((w, i) => <span key={i}>{w}</span>)}</Stack>
                </Banner>
              )}
            </div>
            <div className="stac-ref__list">
              {counts.distances === 0 && counts.planes === 0 && !loading && (
                <EmptyState compact title={t('reference.noItems')} description={t('reference.noItemsHint')} />
              )}
              {counts.distances > 0 && <div className="stac-ref__group">{t('reference.distances', { n: counts.distances })}</div>}
              {doc.distances.map(d => (
                <button key={d.id} type="button" className={`stac-ref__item ${selectedId === d.id ? 'stac-ref__item--selected' : ''}`.trim()} onClick={() => selectDistance(d)}>
                  <span className="stac-ref__item-id">{d.id}</span>
                  <span className="stac-ref__item-main">{d.value_m > 0 ? fmt.lengthText(d.value_m) : t('status.dash')}</span>
                  <Badge size="sm" tone={d.role === 'gauge' ? 'brand' : 'measure'}>{d.role}</Badge>
                  <span className="stac-ref__item-meta">{t('reference.itemMeta', { tool: toolLabel(d.tool), a: d.ends[0].observations.length, b: d.ends[1].observations.length })}</span>
                </button>
              ))}
              {counts.planes > 0 && <div className="stac-ref__group">{t('reference.planes', { n: counts.planes })}</div>}
              {doc.planes.map(p => (
                <button key={p.id} type="button" className={`stac-ref__item ${selectedId === p.id ? 'stac-ref__item--selected' : ''}`.trim()} onClick={() => selectPlane(p)}>
                  <span className="stac-ref__item-id">{p.id}</span>
                  <span className="stac-ref__item-main">{p.label}</span>
                  <span className="stac-ref__item-meta">{t('reference.polygons', { n: p.polygons.length })}</span>
                </button>
              ))}
            </div>
            {selectedDistance && renderDistanceForm(selectedDistance)}
            {selectedPlane && renderPlaneForm(selectedPlane)}
          </aside>

          <section className="stac-ref__main">
            <div className="stac-ref__toolbar">
              <IconButton size="sm" label={t('reference.prevKeyframe')} shortcut="Left" icon={<ChevronLeft aria-hidden />} disabled={kfIndex <= 0} onClick={() => setKfIndex(i => Math.max(0, i - 1))} />
              <span className="stac-ref__nav">{keyframes.length ? t('reference.keyframe', { index: kfIndex + 1, total: keyframes.length }) : t('status.dash')}</span>
              <IconButton size="sm" label={t('reference.nextKeyframe')} shortcut="Right" icon={<ChevronRight aria-hidden />} disabled={kfIndex >= keyframes.length - 1} onClick={() => setKfIndex(i => Math.min(keyframes.length - 1, i + 1))} />
              <span className="stac-ref__filename stac-mono">{currentFile ?? ''}</span>
              <span className="stac-ref__spacer" />
              {pendingSomething && <Button size="sm" variant="ghost" icon={<X aria-hidden />} onClick={cancelPending}>{t('reference.cancelPending')}</Button>}
              {mode === 'plane' && polygon.length > 0 && (
                <>
                  <IconButton size="sm" label={t('reference.undoVertex')} shortcut="Backspace" icon={<Undo2 aria-hidden />} onClick={() => setPolygon(p => p.slice(0, -1))} />
                  <Button size="sm" variant="primary" disabled={polygon.length < 3} onClick={closePolygon}>{t('reference.closePolygon', { n: polygon.length })}</Button>
                </>
              )}
              <IconButton size="sm" label={t('reference.zoomOut')} icon={<ZoomOut aria-hidden />} onClick={() => setZoom(z => Math.max(ZOOM_MIN, z - ZOOM_STEP))} />
              <span className="stac-ref__zoom stac-mono">{fmt.percent(zoom)}</span>
              <IconButton size="sm" label={t('reference.zoomIn')} icon={<ZoomIn aria-hidden />} onClick={() => setZoom(z => Math.min(ZOOM_MAX, z + ZOOM_STEP))} />
              <IconButton size="sm" label={t('reference.resetView')} icon={<Maximize aria-hidden />} onClick={resetView} />
            </div>

            <div ref={containerRef} className={`stac-ref__canvas ${isPanning ? 'stac-ref__canvas--panning' : 'stac-ref__canvas--crosshair'} ${loading ? 'stac-ref__canvas--loading' : ''}`.trim()}
              onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
              {currentFile ? (
                <div className="stac-ref__transform" style={{ '--pan-x': `${pan.x}px`, '--pan-y': `${pan.y}px`, '--zoom': zoom } as React.CSSProperties}>
                  <div className="stac-ref__image-wrap">
                    <img ref={imgRef} src={`/api/sessions/${sessionId}/frames/${currentFile}`} className="stac-ref__image" alt={t('reference.keyframeAlt', { name: currentFile })} draggable={false} onLoad={onImageLoad} />
                    <canvas ref={overlayRef} className="stac-ref__overlay" onClick={handleCanvasClick} onDoubleClick={handleCanvasDoubleClick} onContextMenu={e => e.preventDefault()} />
                  </div>
                </div>
              ) : loading ? null : <EmptyState title={t('reference.noKeyframes')} />}
            </div>

            <div className="stac-ref__hint">
              <span className="stac-ref__hint-text">{hint}</span>
              <span className="stac-ref__hint-keys">{t('reference.hintPan')}</span>
            </div>

            <div className="stac-ref__frames">
              {keyframes.map((kf, i) => (
                <button key={kf} type="button" ref={el => { thumbRefs.current[i] = el }}
                  className={`stac-ref__thumb ${kfIndex === i ? 'stac-ref__thumb--current' : ''}`.trim()}
                  aria-label={t('reference.keyframeAlt', { name: kf })} aria-current={kfIndex === i || undefined} onClick={() => setKfIndex(i)}>
                  <img className="stac-ref__thumb-img" src={`/api/sessions/${sessionId}/frames/${kf}`} loading="lazy" alt="" draggable={false} />
                  {framesWithData.has(parseInt(kf, 10)) && <span className="stac-ref__thumb-dot" title={t('reference.frameHasData')} />}
                  <span className="stac-ref__thumb-idx stac-mono">{i + 1}</span>
                </button>
              ))}
            </div>
          </section>
        </div>
      </Dialog>
      <ConfirmDialog open={discardOpen} variant="danger" title={t('reference.discardTitle')} message={t('reference.discardMessage')}
        confirmLabel={t('reference.discard')} onConfirm={() => { setDiscardOpen(false); onClose() }} onCancel={() => setDiscardOpen(false)} />
    </>
  )
}
