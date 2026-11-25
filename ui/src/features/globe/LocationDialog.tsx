/**
 * THE LOCATION OF A WORK (USER 2026-11-24: "para asignarle a cada sesión, debe abrir un modal donde le puedo cargar
 * la información si no la tiene del video, si la tiene la muestra y permite ajustarla, en el mapa, si no tiene nada,
 * seleccionamos en el mapa donde está la obra a mano. guardamos y ya queda indicado que ese proyecto tiene asignado
 * el punto gps" / "necesita el punto gps real y la orientación verdad?" / and, having tried it: "para ubicar el punto
 * GPS agrega varias herramientas para buscar ciudad, por ejemplo, como google maps tiene, de hecho quise agregar este
 * punto 19°23'53.8"N 99°11'58.4"W y me fue imposible, debe ser mas amigable … sobre todo que la ubicacion sea
 * absolutamente amigable para el usuario").
 *
 * THREE STEPS ON SCREEN: 1 find the work — a name, an address, coordinates in any form (PlaceSearch), the device's
 * own position, the video's GPS — or a touch on the map; 2 lay the plan on it — the scan's published cloud from above
 * (server geo.plan_raster), DRAGGED with the mouse or a finger, TURNED with the slider or ±1° / ±5°; or by 2 points: a
 * point of the plan, the same point on the map, twice (the place, the heading, and the plan-to-map scale shown as a
 * check); 3 save. The map is the globe's (CesiumJS, Esri's public imagery, relief and labels — no account).
 * The plan is turned HERE, on a canvas, then laid north-up: the heading saved is the one seen.
 * Saved: the scan's ORIGIN (lat, lon, the relief's height there) and the heading, with where they came from.
 */
import { useCallback, useEffect, useRef, useState, type MouseEvent } from 'react'
import type * as CesiumNS from 'cesium'
import 'cesium/Build/Cesium/Widgets/widgets.css'
import { Copy, MapPin, RotateCcw, RotateCw } from 'lucide-react'
import { Button } from '../../components/ui/Button'
import { Dialog } from '../../components/ui/Dialog'
import { IconButton } from '../../components/ui/IconButton'
import { useT } from '../../i18n'
import { saveGeoref, type Georef, type GeoSource } from './geo'
import { PlaceSearch } from './PlaceSearch'
import { formatDms, type Place } from './places'
import './location.css'

const ESRI_IMAGERY = 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
const ESRI_RELIEF = 'https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer'
const ESRI_LABELS = ['https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer',
  'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer']
const PIN = '#f28b34'

interface Plan { image: string; cell_m: number; x_min: number; x_max: number; y_min: number; y_max: number; width: number; height: number; scan: string }
interface VideoPlace { lat: number; lon: number; altitude_m?: number; accuracy_m?: number | null }
type LatLon = { lat: number; lon: number }
type PlanPt = { x: number; y: number }

// ── small offsets on the WGS84 ellipsoid (a site: metres to hundreds of metres) ──────────────────────────────────
const A = 6378137, E2 = 6.69437999014e-3, D = Math.PI / 180
function radii(lat: number) { const s = Math.sin(lat * D), w = 1 - E2 * s * s; return { M: A * (1 - E2) / Math.pow(w, 1.5), N: A / Math.sqrt(w) } }
function offset(p: LatLon, e: number, n: number): LatLon {
  const { M, N } = radii(p.lat)
  return { lat: p.lat + n / M / D, lon: p.lon + e / (N * Math.cos(p.lat * D)) / D }
}
function enuOf(from: LatLon, to: LatLon) {
  const { M, N } = radii(from.lat)
  return { e: (to.lon - from.lon) * D * N * Math.cos(from.lat * D), n: (to.lat - from.lat) * D * M }
}
/** a point of the plan (X, Y: the scan written Z-up) -> east, north — server geo.plan_to_enu */
function planToEnu(x: number, y: number, h: number) {
  const r = h * D
  return { e: x * Math.cos(r) + y * Math.sin(r), n: -x * Math.sin(r) + y * Math.cos(r) }
}
const norm360 = (h: number) => ((h % 360) + 360) % 360

export interface LocationDialogProps {
  sessionId: string | null
  token: string | null
  onClose: () => void
  onSaved: (georef: Georef) => void
}

export function LocationDialog({ sessionId, token, onClose, onSaved }: LocationDialogProps) {
  const t = useT()
  const host = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<CesiumNS.Viewer | null>(null)
  const cesiumRef = useRef<typeof CesiumNS | null>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const [imgReady, setImgReady] = useState(0)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [plan, setPlan] = useState<Plan | null>(null)
  const [video, setVideo] = useState<VideoPlace | null | undefined>(undefined)
  const [center, setCenter] = useState<LatLon | null>(null)
  const [heading, setHeading] = useState(0)
  const [source, setSource] = useState<GeoSource>('manual')
  const [mode, setMode] = useState<'hand' | 'points'>('hand')
  const [pts, setPts] = useState<{ p1?: PlanPt; m1?: LatLon; p2?: PlanPt; m2?: LatLon }>({})
  const [scale, setScale] = useState<number | null>(null)
  const [saving, setSaving] = useState(false)
  const [copied, setCopied] = useState(false)
  const modeRef = useRef(mode)
  modeRef.current = mode
  const ptsRef = useRef(pts)
  ptsRef.current = pts
  const centerRef = useRef(center)
  centerRef.current = center
  // the plan on the map: its rectangle and height read by the map every frame (a drag moves it without a rebuild)
  const planGeo = useRef<{ rect?: CesiumNS.Rectangle; height: number }>({ height: 0 })

  const cx = plan ? (plan.x_min + plan.x_max) / 2 : 0
  const cy = plan ? (plan.y_min + plan.y_max) / 2 : 0
  /** where the scan's origin stands: the plan's centre less the centre's own offset, turned */
  const originOf = useCallback((c: LatLon | null, h: number): LatLon | null => {
    if (!c) return null
    const o = planToEnu(-cx, -cy, h)
    return offset(c, o.e, o.n)
  }, [cx, cy])
  const originNow = originOf(center, heading)

  // what the work already has, its plan, its video's GPS
  useEffect(() => {
    if (!sessionId) return
    let gone = false
    setPlan(null); setVideo(undefined); setCenter(null); setHeading(0); setSource('manual'); setPts({}); setScale(null); setError(null)
    imgRef.current = null
    void (async () => {
      try {
        const sid = encodeURIComponent(sessionId)
        const [g, p, v] = await Promise.all([
          fetch(`/api/geo/${sid}`).then(r => r.json()),
          fetch(`/api/geo/${sid}/plan`).then(async r => { const d = await r.json(); if (!r.ok) throw new Error(d.detail ?? `HTTP ${r.status}`); return d }),
          fetch(`/api/geo/${sid}/video`).then(r => r.json()),
        ])
        if (gone) return
        const pl = p as Plan
        setPlan(pl)
        const img = new Image()
        img.onload = () => { imgRef.current = img; setImgReady(n => n + 1) }
        img.src = pl.image
        setVideo((v?.location as VideoPlace | null) ?? null)
        const pcx = (pl.x_min + pl.x_max) / 2, pcy = (pl.y_min + pl.y_max) / 2
        const at = (o: LatLon, h: number) => { const d = planToEnu(pcx, pcy, h); return offset(o, d.e, d.n) }
        const georef = g?.georef as Georef | null
        if (georef) {
          setHeading(georef.heading_deg); setSource(georef.source)
          setCenter(at({ lat: georef.lat, lon: georef.lon }, georef.heading_deg))
        } else if (v?.location) {
          setSource('video')
          setCenter(at({ lat: v.location.lat, lon: v.location.lon }, 0))
        }
      } catch (e) {
        if (!gone) setError(e instanceof Error ? e.message : String(e))
      }
    })()
    return () => { gone = true }
  }, [sessionId])

  // THE MAP, while the window is open: imagery, relief, places and roads; a touch places the work, a drag moves its plan
  useEffect(() => {
    if (!sessionId) return
    let gone = false
    let offWindow = () => {}
    ;(window as unknown as { CESIUM_BASE_URL: string }).CESIUM_BASE_URL = '/cesium/'
    void (async () => {
      try {
        const C = await import('cesium')
        if (gone || !host.current) return
        cesiumRef.current = C
        C.Ion.defaultAccessToken = ''
        ;(C.ArcGisMapService as unknown as { defaultAccessToken?: string }).defaultAccessToken = undefined
        const [relief, imagery] = await Promise.all([
          C.ArcGISTiledElevationTerrainProvider.fromUrl(ESRI_RELIEF), C.ArcGisMapServerImageryProvider.fromUrl(ESRI_IMAGERY),
        ])
        if (gone || !host.current) return
        const v = new C.Viewer(host.current, {
          baseLayer: new C.ImageryLayer(imagery), terrainProvider: relief, baseLayerPicker: false, geocoder: false,
          homeButton: false, sceneModePicker: false, navigationHelpButton: false, animation: false, timeline: false,
          fullscreenButton: false, infoBox: false, selectionIndicator: false, requestRenderMode: true,
          maximumRenderTimeChange: Infinity,
        })
        v.scene.globe.depthTestAgainstTerrain = true
        viewerRef.current = v
        for (const url of ESRI_LABELS) {             // the names of places and the roads over the imagery
          C.ArcGisMapServerImageryProvider.fromUrl(url).then(p => { if (!gone) v.imageryLayers.addImageryProvider(p) }).catch(() => undefined)
        }
        const ground = (pos: CesiumNS.Cartesian2): LatLon | null => {
          const ray = v.camera.getPickRay(pos)
          const at = ray ? v.scene.globe.pick(ray, v.scene) : undefined
          if (!at) return null
          const c = C.Cartographic.fromCartesian(at)
          return { lat: c.latitude / D, lon: c.longitude / D }
        }
        // a DRAG of the plan moves it (the map does not pan meanwhile)
        let drag: { from: LatLon; start: LatLon; moved: boolean } | null = null
        let dragged = false
        const ev = new C.ScreenSpaceEventHandler(v.scene.canvas)
        ev.setInputAction((e: { position: CesiumNS.Cartesian2 }) => {
          if (modeRef.current !== 'hand' || !centerRef.current) return
          const picked = v.scene.pick(e.position) as { id?: { id?: unknown } } | undefined
          if (picked?.id?.id !== 'plan' && picked?.id?.id !== 'origin') return
          const at = ground(e.position)
          if (!at) return
          drag = { from: at, start: centerRef.current, moved: false }
          v.scene.screenSpaceCameraController.enableInputs = false
        }, C.ScreenSpaceEventType.LEFT_DOWN)
        ev.setInputAction((e: { endPosition: CesiumNS.Cartesian2 }) => {
          if (!drag) return
          const at = ground(e.endPosition)
          if (!at) return
          const d = enuOf(drag.from, at)
          drag.moved = true
          setCenter(offset(drag.start, d.e, d.n)); setSource('manual')
        }, C.ScreenSpaceEventType.MOUSE_MOVE)
        const endDrag = () => {
          if (!drag) return
          dragged = drag.moved
          drag = null
          v.scene.screenSpaceCameraController.enableInputs = true
        }
        ev.setInputAction(endDrag, C.ScreenSpaceEventType.LEFT_UP)
        window.addEventListener('pointerup', endDrag)        // released outside the map: the map pans again
        offWindow = () => window.removeEventListener('pointerup', endDrag)
        ev.setInputAction((e: { position: CesiumNS.Cartesian2 }) => {
          if (dragged) { dragged = false; return }
          const here = ground(e.position)
          if (!here) return
          if (modeRef.current === 'hand') { setCenter(here); setSource('manual'); return }
          const p = ptsRef.current            // two points: a map point after its plan point
          if (p.p1 && !p.m1) setPts({ ...p, m1: here })
          else if (p.p2 && !p.m2) setPts({ ...p, m2: here })
        }, C.ScreenSpaceEventType.LEFT_CLICK)
        setReady(true)
      } catch (e) {
        if (!gone) setError(e instanceof Error ? e.message : String(e))
      }
    })()
    return () => { gone = true; offWindow(); setReady(false); viewerRef.current?.destroy(); viewerRef.current = null }
  }, [sessionId])

  // the camera: on the work when it has a place, once
  const flown = useRef(false)
  useEffect(() => { flown.current = false }, [sessionId])
  useEffect(() => {
    const v = viewerRef.current, C = cesiumRef.current
    if (!ready || !v || !C || !center || flown.current) return
    flown.current = true
    v.camera.flyTo({ destination: C.Cartesian3.fromDegrees(center.lon, center.lat, 220), duration: 0 })
  }, [ready, center])

  // a place found: the map goes there; a precise one (an address, a point, coordinates) puts the work there too
  const goTo = (p: Place) => {
    const v = viewerRef.current, C = cesiumRef.current
    const wide = p.extent ? enuOf({ lat: p.extent.south, lon: p.extent.west }, { lat: p.extent.north, lon: p.extent.east }) : null
    const broad = !!wide && Math.hypot(wide.e, wide.n) > 1500       // a city, a region: the user touches the spot
    if (v && C) {
      flown.current = true
      if (broad && p.extent) v.camera.flyTo({ destination: C.Rectangle.fromDegrees(p.extent.west, p.extent.south, p.extent.east, p.extent.north), duration: 1.5 })
      else v.camera.flyTo({ destination: C.Cartesian3.fromDegrees(p.lon, p.lat, 260), duration: 1.5 })
    }
    if (!broad) { setCenter({ lat: p.lat, lon: p.lon }); setSource('manual'); setMode('hand') }
  }

  // two points: the place and the heading from P1 → M1, P2 → M2 (scale shown, never applied)
  useEffect(() => {
    const { p1, m1, p2, m2 } = pts
    if (!p1 || !m1 || !p2 || !m2) return
    const w = enuOf(m1, m2), vx = p2.x - p1.x, vy = p2.y - p1.y
    const lv = Math.hypot(vx, vy), lw = Math.hypot(w.e, w.n)
    if (lv < 1e-6 || lw < 1e-6) { setError(t('location.pointsTooClose')); return }
    const h = norm360((Math.atan2(w.e, w.n) - Math.atan2(vx, vy)) / D)
    const o1 = planToEnu(p1.x, p1.y, h)
    const o = offset(m1, -o1.e, -o1.n)                          // the origin
    const c = planToEnu(cx, cy, h)
    setHeading(Number(h.toFixed(2))); setCenter(offset(o, c.e, c.n)); setSource('points'); setScale(lw / lv)
  }, [pts, cx, cy, t])

  // THE PLAN ON THE MAP — the entity once; its rectangle and height follow the centre and the heading every frame;
  // its picture, turned on a canvas, is made again only when the heading changes
  useEffect(() => {
    const v = viewerRef.current, C = cesiumRef.current
    if (!ready || !v || !C || !plan) return
    if (!v.entities.getById('plan')) {
      v.entities.add({
        id: 'plan',
        rectangle: {
          coordinates: new C.CallbackProperty(() => planGeo.current.rect, false) as unknown as CesiumNS.Property,
          height: new C.CallbackProperty(() => planGeo.current.height, false) as unknown as CesiumNS.Property,
        },
      })
    }
  }, [ready, plan])
  useEffect(() => {
    const v = viewerRef.current, C = cesiumRef.current
    if (!ready || !v || !C || !plan) return
    const ent = v.entities.getById('plan')
    if (!center) { planGeo.current.rect = undefined; v.scene.requestRender(); return }
    const r = heading * D, cos = Math.abs(Math.cos(r)), sin = Math.abs(Math.sin(r))
    const W = Math.ceil(cos * plan.width + sin * plan.height), H = Math.ceil(sin * plan.width + cos * plan.height)
    const hx = W * plan.cell_m / 2, hy = H * plan.cell_m / 2
    const sw = offset(center, -hx, -hy), ne = offset(center, hx, hy)
    planGeo.current.rect = C.Rectangle.fromDegrees(sw.lon, sw.lat, ne.lon, ne.lat)
    const g = v.scene.globe.getHeight(C.Cartographic.fromDegrees(center.lon, center.lat))
    planGeo.current.height = (g ?? 0) + 0.3
    if (ent?.rectangle) ent.show = true
    v.scene.requestRender()
  }, [ready, plan, center, heading])
  useEffect(() => {
    const v = viewerRef.current, C = cesiumRef.current, img = imgRef.current
    const ent = v?.entities.getById('plan')
    if (!ready || !v || !C || !plan || !img || !ent?.rectangle) return
    const r = heading * D, cos = Math.abs(Math.cos(r)), sin = Math.abs(Math.sin(r))
    const W = Math.ceil(cos * plan.width + sin * plan.height), H = Math.ceil(sin * plan.width + cos * plan.height)
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, W); canvas.height = Math.max(1, H)
    const g = canvas.getContext('2d')
    if (g) {
      g.translate(W / 2, H / 2)
      g.rotate(r)                                     // clockwise on the screen = clockwise from north
      g.drawImage(img, -plan.width / 2, -plan.height / 2)
    }
    ent.rectangle.material = new C.ImageMaterialProperty({ image: canvas, transparent: true, color: C.Color.WHITE.withAlpha(0.8) })
    v.scene.requestRender()
  }, [ready, plan, heading, imgReady])
  // the origin and the two points' marks
  useEffect(() => {
    const v = viewerRef.current, C = cesiumRef.current
    if (!ready || !v || !C) return
    for (const id of ['origin', 'm1', 'm2']) { const e = v.entities.getById(id); if (e) v.entities.remove(e) }
    const mark = (id: string, p: LatLon | null | undefined, size: number, label?: string) => {
      if (!p) return
      v.entities.add({
        id, position: C.Cartesian3.fromDegrees(p.lon, p.lat),
        point: { pixelSize: size, color: C.Color.fromCssColorString(PIN), outlineColor: C.Color.WHITE, outlineWidth: 2,
                 heightReference: C.HeightReference.CLAMP_TO_GROUND, disableDepthTestDistance: Number.POSITIVE_INFINITY },
        label: label ? { text: label, font: '600 13px sans-serif', fillColor: C.Color.WHITE, outlineColor: C.Color.BLACK,
                         outlineWidth: 3, style: C.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new C.Cartesian2(0, -16),
                         heightReference: C.HeightReference.CLAMP_TO_GROUND, disableDepthTestDistance: Number.POSITIVE_INFINITY } : undefined,
      })
    }
    mark('origin', originNow, 9)
    mark('m1', pts.m1, 12, '1'); mark('m2', pts.m2, 12, '2')
    v.scene.requestRender()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, center, heading, pts, cx, cy])

  // a click on the plan (two points): where on the plan, in metres
  const onPlanClick = (e: MouseEvent<HTMLImageElement>) => {
    if (!plan || mode !== 'points') return
    const box = e.currentTarget.getBoundingClientRect()
    const x = plan.x_min + (e.clientX - box.left) / box.width * (plan.x_max - plan.x_min)
    const y = plan.y_max - (e.clientY - box.top) / box.height * (plan.y_max - plan.y_min)
    setPts(p => {
      if (!p.p1 || !p.m1) return { p1: { x, y } }                       // the first plan point (again)
      if (!p.p2 || !p.m2) return { p1: p.p1, m1: p.m1, p2: { x, y } }   // the second one (again)
      return { p1: { x, y } }                                           // all four given: a new pair
    })
    setScale(null)
  }
  const step = !pts.p1 ? 'p1' : !pts.m1 ? 'm1' : !pts.p2 ? 'p2' : !pts.m2 ? 'm2' : 'done'
  const planDot = (p?: PlanPt) => (plan && p ? { left: `${(p.x - plan.x_min) / (plan.x_max - plan.x_min) * 100}%`, top: `${(plan.y_max - p.y) / (plan.y_max - plan.y_min) * 100}%` } : undefined)

  const save = async () => {
    const C = cesiumRef.current, v = viewerRef.current, o = originNow
    if (!sessionId || !o || !plan) return
    setSaving(true); setError(null)
    try {
      let height = 0
      if (C && v) {
        const [c] = await C.sampleTerrainMostDetailed(v.terrainProvider, [C.Cartographic.fromDegrees(o.lon, o.lat)])
        height = Number.isFinite(c?.height) ? c.height : 0
      }
      const georef = await saveGeoref(sessionId, {
        lat: o.lat, lon: o.lon, height_m: height, heading_deg: heading, source, scan: plan.scan,
        accuracy_m: source === 'video' ? video?.accuracy_m ?? null : null,
      }, token)
      onSaved(georef)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const turn = (d: number) => { setHeading(h => Number(norm360(h + d).toFixed(2))); if (source !== 'manual') setSource('manual') }
  const copy = (text: string) => {
    void navigator.clipboard?.writeText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) }).catch(() => undefined)
  }

  return (
    <Dialog open={!!sessionId} size="full" icon={<MapPin aria-hidden />} title={t('location.title', { name: sessionId ?? '' })}
      subtitle={t('location.subtitle')} onClose={onClose} busy={saving}
      footer={(
        <>
          {!token && <span className="stac-location__note">{t('globe.signIn')}</span>}
          <Button variant="ghost" onClick={onClose} disabled={saving}>{t('globe.cancel')}</Button>
          <Button variant="primary" loading={saving} disabled={!token || !originNow || !plan} onClick={() => void save()}>{t('globe.save')}</Button>
        </>
      )}>
      <div className="stac-location">
        <div className="stac-location__side">
          <section className="stac-location__step">
            <h3><span>1</span> {t('location.step1')}</h3>
            <PlaceSearch onPick={goTo} autoFocus />
            {video === undefined ? <p className="stac-location__note">{t('location.reading')}</p>
              : video && (
                <div className="stac-location__video">
                  <p>{t('location.videoHas', { place: formatDms(video.lat, video.lon) })}
                    {video.accuracy_m != null ? ` (±${Math.round(video.accuracy_m)} m)` : ''}</p>
                  <Button size="sm" variant="secondary" onClick={() => {
                    const c = planToEnu(cx, cy, heading); setCenter(offset({ lat: video.lat, lon: video.lon }, c.e, c.n)); setSource('video')
                    goTo({ lat: video.lat, lon: video.lon, label: '' })
                  }}>{t('location.useVideo')}</Button>
                </div>
              )}
            <p className="stac-location__note">{t('location.step1Hint')}</p>
          </section>

          <section className="stac-location__step">
            <h3><span>2</span> {t('location.step2')}</h3>
            <div className="stac-location__modes" role="tablist">
              <Button size="sm" variant={mode === 'hand' ? 'primary' : 'ghost'} onClick={() => setMode('hand')}>{t('location.byHand')}</Button>
              <Button size="sm" variant={mode === 'points' ? 'primary' : 'ghost'} onClick={() => { setMode('points'); setPts({}); setScale(null) }}>{t('location.byPoints')}</Button>
            </div>
            {mode === 'hand' ? (
              <>
                <p className="stac-location__note">{t('location.handHint')}</p>
                <label className="stac-location__label" htmlFor="stac-location-heading">{t('location.heading')}</label>
                <div className="stac-location__heading">
                  <Button size="sm" variant="ghost" icon={<RotateCcw aria-hidden />} onClick={() => turn(-5)}>5°</Button>
                  <Button size="sm" variant="ghost" onClick={() => turn(-1)}>−1°</Button>
                  <input id="stac-location-heading" type="number" min={0} max={359.99} step={0.1} value={heading}
                    onChange={e => { const n = Number(e.target.value); if (Number.isFinite(n)) { setHeading(norm360(n)); setSource('manual') } }} />
                  <Button size="sm" variant="ghost" onClick={() => turn(1)}>+1°</Button>
                  <Button size="sm" variant="ghost" icon={<RotateCw aria-hidden />} onClick={() => turn(5)}>5°</Button>
                </div>
                <input className="stac-location__slider" type="range" min={0} max={359.9} step={0.1} value={heading}
                  aria-label={t('location.heading')} onChange={e => { setHeading(Number(e.target.value)); setSource('manual') }} />
              </>
            ) : (
              <>
                <p className="stac-location__note">{t(`location.step_${step}`)}</p>
                {plan && (
                  <div className="stac-location__plan">
                    <img src={plan.image} alt={t('location.planAlt')} onClick={onPlanClick} draggable={false} />
                    {pts.p1 && <span className="stac-location__dot" style={planDot(pts.p1)}>1</span>}
                    {pts.p2 && <span className="stac-location__dot" style={planDot(pts.p2)}>2</span>}
                  </div>
                )}
                {scale != null && <p className="stac-location__note">{t('location.scale', { s: scale.toFixed(3) })}</p>}
              </>
            )}
          </section>

          <section className="stac-location__step">
            <h3><span>3</span> {t('location.step3')}</h3>
            <div className="stac-location__result">
              {originNow ? <>
                <div className="stac-location__coords">
                  <span>{formatDms(originNow.lat, originNow.lon)}</span>
                  <IconButton size="sm" label={copied ? t('location.copied') : t('location.copy')} icon={<Copy aria-hidden />}
                    onClick={() => copy(formatDms(originNow.lat, originNow.lon))} />
                </div>
                <div className="stac-location__decimal">{originNow.lat.toFixed(6)}, {originNow.lon.toFixed(6)}</div>
                <div>{t('location.headingIs', { h: heading.toFixed(1) })} · {t(`globe.source_${source}`)}</div>
              </> : <div className="stac-location__note">{t('location.nothingYet')}</div>}
            </div>
          </section>
          {error && <p className="stac-location__error" role="alert">{error}</p>}
        </div>
        <div ref={host} className="stac-location__map">
          {!ready && !error && <div className="stac-location__loading">{t('globe.loading')}</div>}
        </div>
      </div>
    </Dialog>
  )
}
