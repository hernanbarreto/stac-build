/**
 * THE GLOBE — drawn over the viewer's area while the left panel's globe section is open (USER 2026-11-24: "cuando
 * seleccionamos el icono del mundo, muestra cesium con todos los pines donde están los proyectos, cuando selecciono
 * alguno de los pines, me lleva a ese lugar del mapa cesium y abre la sesión por defecto" / "cesium siempre sin
 * cuenta, baja la mejor calidad que se pueda tener … veamos de conseguir relieves sí o sí").
 *
 * CesiumJS from the interface's own node_modules (its workers and assets served at /cesium/, public/cesium), NO
 * Cesium ion and NO ArcGIS key: the imagery is Esri's public World Imagery and the relief Esri's public World
 * Elevation 3D (both answered with no account, 2026-11-24), their credit shown by Cesium. A pin per placed work; a
 * click on one flies there and opens its session. Placing a work: a click on the relief, then Save.
 * The 3D viewer underneath stays mounted: closing the globe gives it back as it was.
 */
import { useEffect, useRef, useState } from 'react'
import type * as CesiumNS from 'cesium'
import 'cesium/Build/Cesium/Widgets/widgets.css'
import { Button } from '../../components/ui/Button'
import { useT } from '../../i18n'
import { formatPlace, type GeoProject } from './geo'
import './globe.css'

const ESRI_IMAGERY = 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
const ESRI_RELIEF = 'https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer'
const PIN = '#f28b34'                       // --brand-orange
const CANDIDATE = '__stac_candidate__'

export interface GlobeViewProps {
  projects: GeoProject[]
  /** the work being placed (its session id), or null */
  placing: string | null
  /** fly to this work (n changes for every request) */
  focus: { sid: string; n: number } | null
  onPlaced: (sid: string, place: { lat: number; lon: number; height_m: number }) => Promise<void>
  onCancelPlacing: () => void
  onOpen: (sid: string) => void
}

type Cesium = typeof CesiumNS

export default function GlobeView({ projects, placing, focus, onPlaced, onCancelPlacing, onOpen }: GlobeViewProps) {
  const t = useT()
  const host = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<CesiumNS.Viewer | null>(null)
  const cesiumRef = useRef<Cesium | null>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [candidate, setCandidate] = useState<{ lat: number; lon: number; height_m: number } | null>(null)
  const [saving, setSaving] = useState(false)
  const placingRef = useRef(placing)
  placingRef.current = placing
  const projectsRef = useRef(projects)
  projectsRef.current = projects
  const onOpenRef = useRef(onOpen)
  onOpenRef.current = onOpen
  const framedRef = useRef(false)

  // THE VIEWER, once
  useEffect(() => {
    let gone = false
    ;(window as unknown as { CESIUM_BASE_URL: string }).CESIUM_BASE_URL = '/cesium/'
    void (async () => {
      try {
        const C: Cesium = await import('cesium')
        if (gone || !host.current) return
        cesiumRef.current = C
        C.Ion.defaultAccessToken = ''                                              // no Cesium ion: no account
        ;(C.ArcGisMapService as unknown as { defaultAccessToken?: string }).defaultAccessToken = undefined // no key
        const [relief, imagery] = await Promise.all([
          C.ArcGISTiledElevationTerrainProvider.fromUrl(ESRI_RELIEF),
          C.ArcGisMapServerImageryProvider.fromUrl(ESRI_IMAGERY),
        ])
        if (gone || !host.current) return
        const viewer = new C.Viewer(host.current, {
          baseLayer: new C.ImageryLayer(imagery), terrainProvider: relief,
          baseLayerPicker: false, geocoder: false, homeButton: true, sceneModePicker: false, navigationHelpButton: false,
          animation: false, timeline: false, fullscreenButton: false, infoBox: false, selectionIndicator: false,
          requestRenderMode: true, maximumRenderTimeChange: Infinity,
        })
        viewer.scene.globe.depthTestAgainstTerrain = true
        viewer.scene.globe.enableLighting = false
        viewerRef.current = viewer
        const clicks = new C.ScreenSpaceEventHandler(viewer.scene.canvas)
        clicks.setInputAction((e: { position: CesiumNS.Cartesian2 }) => {
          const v = viewerRef.current
          if (!v) return
          if (placingRef.current) {
            // where the work stands: the point of the relief under the click
            const ray = v.camera.getPickRay(e.position)
            const at = ray ? v.scene.globe.pick(ray, v.scene) : undefined
            if (!at) return
            const c = C.Cartographic.fromCartesian(at)
            setCandidate({ lat: C.Math.toDegrees(c.latitude), lon: C.Math.toDegrees(c.longitude), height_m: c.height })
            return
          }
          const picked = v.scene.pick(e.position) as { id?: { id?: unknown } } | undefined
          const sid = typeof picked?.id?.id === 'string' ? picked.id.id : null
          if (!sid || !projectsRef.current.some(p => p.session_id === sid && p.georef)) return
          const ent = v.entities.getById(sid)
          if (!ent) return
          void v.flyTo(ent, { duration: 2, offset: new C.HeadingPitchRange(0, C.Math.toRadians(-35), 700) })
            .then(arrived => { if (arrived) onOpenRef.current(sid) })
        }, C.ScreenSpaceEventType.LEFT_CLICK)
        setReady(true)
      } catch (err) {
        if (!gone) setError(err instanceof Error ? err.message : String(err))
      }
    })()
    return () => {
      gone = true
      viewerRef.current?.destroy()
      viewerRef.current = null
    }
  }, [])

  // THE PINS: one per placed work
  useEffect(() => {
    const v = viewerRef.current, C = cesiumRef.current
    if (!ready || !v || !C) return
    for (const e of v.entities.values.slice()) if (e.id !== CANDIDATE) v.entities.remove(e)
    const color = C.Color.fromCssColorString(PIN)
    for (const p of projects) {
      if (!p.georef) continue
      v.entities.add({
        id: p.session_id, name: p.session_id,
        position: C.Cartesian3.fromDegrees(p.georef.lon, p.georef.lat, p.georef.height_m),
        point: { pixelSize: 14, color, outlineColor: C.Color.WHITE, outlineWidth: 2,
                 heightReference: C.HeightReference.CLAMP_TO_GROUND, disableDepthTestDistance: Number.POSITIVE_INFINITY },
        label: { text: p.session_id, font: '600 14px sans-serif', fillColor: C.Color.WHITE, outlineColor: C.Color.BLACK,
                 outlineWidth: 3, style: C.LabelStyle.FILL_AND_OUTLINE, verticalOrigin: C.VerticalOrigin.BOTTOM,
                 pixelOffset: new C.Cartesian2(0, -14), heightReference: C.HeightReference.CLAMP_TO_GROUND,
                 disableDepthTestDistance: Number.POSITIVE_INFINITY },
      })
    }
    if (!framedRef.current && projects.some(p => p.georef)) {
      framedRef.current = true
      void v.flyTo(v.entities, { duration: 0 })
    }
    v.scene.requestRender()
  }, [ready, projects])

  // GO TO A WORK (the list's name)
  useEffect(() => {
    const v = viewerRef.current, C = cesiumRef.current
    if (!ready || !v || !C || !focus) return
    const ent = v.entities.getById(focus.sid)
    if (ent) void v.flyTo(ent, { duration: 2, offset: new C.HeadingPitchRange(0, C.Math.toRadians(-35), 700) })
  }, [ready, focus])

  // PLACING: the candidate's pin; a work placed before is shown where it was
  useEffect(() => { setCandidate(null) }, [placing])
  useEffect(() => {
    const v = viewerRef.current, C = cesiumRef.current
    if (!ready || !v || !C) return
    const old = v.entities.getById(CANDIDATE)
    if (old) v.entities.remove(old)
    if (candidate) {
      v.entities.add({
        id: CANDIDATE,
        position: C.Cartesian3.fromDegrees(candidate.lon, candidate.lat, candidate.height_m),
        point: { pixelSize: 16, color: C.Color.WHITE, outlineColor: C.Color.fromCssColorString(PIN), outlineWidth: 4,
                 disableDepthTestDistance: Number.POSITIVE_INFINITY },
      })
    }
    v.scene.requestRender()
  }, [ready, candidate])

  const placingName = placing ?? ''
  const save = async () => {
    if (!placing || !candidate) return
    setSaving(true)
    try {
      await onPlaced(placing, candidate)
      setCandidate(null)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="stac-globe" aria-label={t('globe.title')}>
      <div ref={host} className="stac-globe__canvas" />
      {!ready && !error && <div className="stac-globe__status">{t('globe.loading')}</div>}
      {error && <div className="stac-globe__status stac-globe__status--error" role="alert">{t('globe.failed', { detail: error })}</div>}
      {placing && ready && (
        <div className="stac-globe__bar" role="status">
          {candidate
            ? <span>{t('globe.candidate', { name: placingName, place: formatPlace(candidate) })}</span>
            : <span>{t('globe.placeHint', { name: placingName })}</span>}
          {candidate && <Button size="sm" variant="primary" loading={saving} onClick={() => void save()}>{t('globe.save')}</Button>}
          <Button size="sm" variant="ghost" disabled={saving} onClick={onCancelPlacing}>{t('globe.cancel')}</Button>
        </div>
      )}
    </div>
  )
}
