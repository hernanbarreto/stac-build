/**
 * THE GLOBE — drawn over the viewer's area while the left panel's globe section is open (USER 2026-11-24: "cuando
 * seleccionamos el icono del mundo, muestra cesium con todos los pines donde están los proyectos, cuando selecciono
 * alguno de los pines, me lleva a ese lugar del mapa cesium y abre la sesión por defecto" / "cesium siempre sin
 * cuenta, baja la mejor calidad que se pueda tener … veamos de conseguir relieves sí o sí").
 *
 * CesiumJS from the interface's own node_modules (its workers and assets served at /cesium/, public/cesium), NO
 * Cesium ion and NO ArcGIS key: the imagery is Esri's public World Imagery and the relief Esri's public World
 * Elevation 3D (both answered with no account, 2026-11-24), their credit shown by Cesium. A pin per placed work; a
 * click on one flies there and opens its session. A work is placed in its Location window (LocationDialog).
 * THE WORK ITSELF stands on the globe where it was placed (USER 2026-11-24: "agrega todo lo que puedas en esta etapa
 * si, malla en el globo, nube si no tiene malla"): its GenRecon mesh's tileset, else a million points of its cloud
 * (server geo_api /content), loaded once when the work is chosen in the list or the camera comes within 5 km.
 * The 3D viewer underneath stays mounted: closing the globe gives it back as it was.
 */
import { useEffect, useRef, useState } from 'react'
import type * as CesiumNS from 'cesium'
import 'cesium/Build/Cesium/Widgets/widgets.css'
import { useT } from '../../i18n'
import type { GeoProject, Georef } from './geo'
import './globe.css'

const ESRI_IMAGERY = 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
const ESRI_RELIEF = 'https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer'
const PIN = '#f28b34'                       // --brand-orange

export interface GlobeViewProps {
  projects: GeoProject[]
  /** fly to this work (n changes for every request) */
  focus: { sid: string; n: number } | null
  onOpen: (sid: string) => void
  /** the works whose IFC is shown (USER 2026-11-24: "o IFC si se puede") */
  ifcOn: string[]
}

type Cesium = typeof CesiumNS
type Floor = { R: number[][]; s: number; t: number[] } | null
interface Content { mesh_tileset: string | null; cloud_tileset: string | null; floor: Floor }
const NEAR_M = 5000                          // the camera this close to a work: its 3D is loaded

/** where a work's 3D stands: the east-north-up frame at its origin, turned by its heading (the plan's +Y to its
 *  azimuth: e = X cos h + Y sin h, n = -X sin h + Y cos h — server geo.plan_to_enu). The mesh's tiles are written in
 *  glTF's Y up and say gltfUpAxis Z, so their turn to Z up is ours, and their floor adjustment too; the cloud's
 *  glTF is turned by the viewer itself and its adjustment was applied by the server. */
function placement(C: Cesium, g: Georef, floor: Floor, ownTurnToZUp: boolean): CesiumNS.Matrix4 {
  const enu = C.Transforms.eastNorthUpToFixedFrame(C.Cartesian3.fromDegrees(g.lon, g.lat, g.height_m))
  const turn = C.Matrix4.fromRotation(C.Matrix3.fromRotationZ(-g.heading_deg * Math.PI / 180))
  let m = C.Matrix4.multiply(enu, turn, new C.Matrix4())
  if (ownTurnToZUp) {
    m = C.Matrix4.multiply(m, C.Matrix4.fromRotation(C.Matrix3.fromRotationX(Math.PI / 2)), new C.Matrix4())   // (x, y, z) -> (x, -z, y)
    if (floor) {
      const { R, s: k, t } = floor
      const rot = new C.Matrix3(k * R[0][0], k * R[0][1], k * R[0][2], k * R[1][0], k * R[1][1], k * R[1][2],
        k * R[2][0], k * R[2][1], k * R[2][2])
      m = C.Matrix4.multiply(m, C.Matrix4.fromRotationTranslation(rot, new C.Cartesian3(t[0], t[1], t[2])), new C.Matrix4())
    }
  }
  return m
}

export default function GlobeView({ projects, focus, onOpen, ifcOn }: GlobeViewProps) {
  const t = useT()
  const host = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<CesiumNS.Viewer | null>(null)
  const cesiumRef = useRef<Cesium | null>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const projectsRef = useRef(projects)
  projectsRef.current = projects
  const onOpenRef = useRef(onOpen)
  onOpenRef.current = onOpen
  const framedRef = useRef(false)
  // the works whose 3D is on the globe (or on its way), and how each one is placed
  const worksRef = useRef(new Map<string, { tileset?: CesiumNS.Cesium3DTileset; content?: Content; ownTurn: boolean }>())
  const [notice, setNotice] = useState<string | null>(null)
  // the works' IFC models on the globe (each IFC: web-ifc as the BIM panel loads it, turned into glTF here)
  const ifcRef = useRef(new Map<string, CesiumNS.Model[]>())
  const ifcOnRef = useRef(ifcOn)
  ifcOnRef.current = ifcOn

  // A WORK'S 3D: its content once, placed by its georeference; placed again when that changes
  const loadWork = useRef<(sid: string) => void>(() => {})
  loadWork.current = (sid: string) => {
    const v = viewerRef.current, C = cesiumRef.current
    const p = projectsRef.current.find(x => x.session_id === sid)
    if (!v || !C || !p?.georef || worksRef.current.has(sid)) return
    const entry: { tileset?: CesiumNS.Cesium3DTileset; content?: Content; ownTurn: boolean } = { ownTurn: false }
    worksRef.current.set(sid, entry)
    void (async () => {
      try {
        const c = await fetch(`/api/geo/${encodeURIComponent(sid)}/content`, { cache: 'no-store' }).then(r => r.json()) as Content
        const url = c.mesh_tileset ?? c.cloud_tileset
        if (!url || !viewerRef.current) return
        const ts = await C.Cesium3DTileset.fromUrl(url, { maximumScreenSpaceError: 8 })
        if (!viewerRef.current || worksRef.current.get(sid) !== entry) { ts.destroy(); return }
        entry.content = c
        entry.ownTurn = !!c.mesh_tileset
        const g = projectsRef.current.find(x => x.session_id === sid)?.georef
        if (!g) { ts.destroy(); return }
        ts.modelMatrix = placement(C, g, c.floor, entry.ownTurn)
        if (!c.mesh_tileset) ts.style = new C.Cesium3DTileStyle({ pointSize: 3 })
        entry.tileset = viewerRef.current.scene.primitives.add(ts)
        viewerRef.current.scene.requestRender()
      } catch (err) {
        worksRef.current.delete(sid)
        setNotice(t('globe.workFailed', { name: sid, detail: err instanceof Error ? err.message : String(err) }))
      }
    })()
  }

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
        viewerRef.current = viewer
        const clicks = new C.ScreenSpaceEventHandler(viewer.scene.canvas)
        clicks.setInputAction((e: { position: CesiumNS.Cartesian2 }) => {
          const v = viewerRef.current
          if (!v) return
          const picked = v.scene.pick(e.position) as { id?: { id?: unknown } } | undefined
          const sid = typeof picked?.id?.id === 'string' ? picked.id.id : null
          if (!sid || !projectsRef.current.some(p => p.session_id === sid && p.georef)) return
          const ent = v.entities.getById(sid)
          if (!ent) return
          void v.flyTo(ent, { duration: 2, offset: new C.HeadingPitchRange(0, C.Math.toRadians(-35), 700) })
            .then(arrived => { if (arrived) onOpenRef.current(sid) })
        }, C.ScreenSpaceEventType.LEFT_CLICK)
        viewer.camera.moveEnd.addEventListener(() => {
          const v = viewerRef.current
          if (!v) return
          for (const p of projectsRef.current) {
            if (!p.georef || worksRef.current.has(p.session_id)) continue
            const at = C.Cartesian3.fromDegrees(p.georef.lon, p.georef.lat, p.georef.height_m)
            if (C.Cartesian3.distance(v.camera.positionWC, at) < NEAR_M) loadWork.current(p.session_id)
          }
        })
        setReady(true)
      } catch (err) {
        if (!gone) setError(err instanceof Error ? err.message : String(err))
      }
    })()
    return () => {
      gone = true
      worksRef.current.clear()
      ifcRef.current.clear()
      viewerRef.current?.destroy()
      viewerRef.current = null
    }
  }, [])

  // THE PINS: one per placed work
  useEffect(() => {
    const v = viewerRef.current, C = cesiumRef.current
    if (!ready || !v || !C) return
    v.entities.removeAll()
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
    for (const [sid, w] of [...worksRef.current]) {
      const g = projects.find(p => p.session_id === sid)?.georef
      if (!g) { if (w.tileset) v.scene.primitives.remove(w.tileset); worksRef.current.delete(sid); continue }
      if (w.tileset && w.content) w.tileset.modelMatrix = placement(C, g, w.content.floor, w.ownTurn)
    }
    for (const [sid, models] of ifcRef.current) {
      const g = projects.find(p => p.session_id === sid)?.georef
      for (const m of models) { if (g) m.modelMatrix = placement(C, g, null, false); else m.show = false }
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
    loadWork.current(focus.sid)
    if (ent) void v.flyTo(ent, { duration: 2, offset: new C.HeadingPitchRange(0, C.Math.toRadians(-35), 120) })
  }, [ready, focus])

  // AN IFC ON THE GLOBE: written from the scan, it is in the display frame (its floor adjustment in it) and glTF's
  // Y up once web-ifc has read it; the viewer turns glTF to Z up: placed as the cloud is
  useEffect(() => {
    const v = viewerRef.current, C = cesiumRef.current
    if (!ready || !v || !C) return
    for (const sid of ifcOn) {
      const p = projects.find(x => x.session_id === sid)
      const have = ifcRef.current.get(sid)
      if (have) { have.forEach(m => { m.show = true }); continue }
      if (!p?.georef || !p.ifcs.length) continue
      const models: CesiumNS.Model[] = []
      ifcRef.current.set(sid, models)
      setNotice(t('globe.ifcLoading', { name: sid }))
      void (async () => {
        try {
          const [{ loadIFC }, { GLTFExporter }] = await Promise.all([
            import('../../components/IFCLoader'), import('three/examples/jsm/exporters/GLTFExporter.js')])
          for (const name of p.ifcs) {
            const r = await loadIFC(`/api/sessions/${encodeURIComponent(sid)}/bim/${encodeURIComponent(name)}`, name)
            const glb = await new GLTFExporter().parseAsync(r.group, { binary: true }) as ArrayBuffer
            r.group.traverse(o => { (o as unknown as { geometry?: { dispose?: () => void } }).geometry?.dispose?.() })
            const url = URL.createObjectURL(new Blob([glb], { type: 'model/gltf-binary' }))
            const g = projectsRef.current.find(x => x.session_id === sid)?.georef
            if (!g || !viewerRef.current || ifcRef.current.get(sid) !== models) { URL.revokeObjectURL(url); return }
            const model = await C.Model.fromGltfAsync({ url, modelMatrix: placement(C, g, null, false) })
            model.readyEvent.addEventListener(() => URL.revokeObjectURL(url))
            model.show = ifcOnRef.current.includes(sid)
            models.push(viewerRef.current.scene.primitives.add(model))
          }
          setNotice(null)
        } catch (err) {
          ifcRef.current.delete(sid)
          setNotice(t('globe.workFailed', { name: sid, detail: err instanceof Error ? err.message : String(err) }))
        } finally {
          viewerRef.current?.scene.requestRender()
        }
      })()
    }
    for (const [sid, models] of ifcRef.current) if (!ifcOn.includes(sid)) models.forEach(m => { m.show = false })
    v.scene.requestRender()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, ifcOn, projects])

  return (
    <div className="stac-globe" aria-label={t('globe.title')}>
      <div ref={host} className="stac-globe__canvas" />
      {!ready && !error && <div className="stac-globe__status">{t('globe.loading')}</div>}
      {error && <div className="stac-globe__status stac-globe__status--error" role="alert">{t('globe.failed', { detail: error })}</div>}
      {notice && <button type="button" className="stac-globe__notice" onClick={() => setNotice(null)}>{notice}</button>}
    </div>
  )
}
