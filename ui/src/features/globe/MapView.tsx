/**
 * THE WORKS ON THE MAP — Mapbox GL JS (USER 2026-11-28, on the first globe's imagery — "las imagenes … son una mierda" —
 * Mapbox chosen; then: "sin muestra de satelite, muestra mas bien sobria con 3d y mas bien
 * oscurito" / "al presionar el mundo en el left bar, debe mostrarme el mundo con los pines de las obras que tengo
 * georeferenciadas, al tocar sobre alguna, debe acercarme y mostrar el glb liviano, en el lugar correspondiente" /
 * "calidad visual super profesional, rapido").
 *
 * Mapbox's Standard style, monochrome at night, its 3D buildings, no satellite and no points of interest; the globe
 * from space, a pin per placed work. A pin (or the work in the list) flies there and stands the work's LIGHT model
 * where it was placed: the root tile of its GenRecon mesh's tileset (the whole mesh simplified, its textures reduced —
 * grupo_electrogeno's: 411 k triangles, 8 MB, against the 560 MB GLB), else a million points of its cloud; its IFC on
 * demand (the root tile of the IFC's tiles). The models are drawn by three.js inside Mapbox's own frame (a custom
 * layer: Mapbox's matrix × the work's placement — the depth shared, so the buildings hide what they stand before).
 * The token: the backend's (GET /api/geo/map_config, MAPBOX_TOKEN in .env).
 */
import { useEffect, useRef, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js'
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js'
import { Cloud, CloudDrizzle, CloudFog, CloudLightning, CloudRain, CloudSnow, CloudSun, Sun } from 'lucide-react'
import { useT } from '../../i18n'
import type { GeoProject, Georef } from './geo'
import { PlaceSearch } from './PlaceSearch'
import type { Place } from './places'
import './globe.css'

export interface MapViewProps {
  projects: GeoProject[]
  /** fly to this work (n changes for every request) */
  focus: { sid: string; n: number } | null
  /** the works whose IFC stands on the map */
  ifcOn: string[]
}

type Floor = { R: number[][]; s: number; t: number[] } | null
interface Weather { sid: string; code: number; temperature_c: number; apparent_c: number; humidity_pct: number; wind_kmh: number
                    wind_dir_deg: number; precip_mm: number; cloud_pct: number; is_day: boolean; time: string }
type Sky = 'clear' | 'partly' | 'cloudy' | 'fog' | 'drizzle' | 'rain' | 'snow' | 'storm'

/** the sky of a WMO weather code (Open-Meteo's), and how hard it falls (0 – 1) */
function skyOf(code: number): { sky: Sky; strength: number } {
  if (code <= 1) return { sky: 'clear', strength: 0 }
  if (code === 2) return { sky: 'partly', strength: 0 }
  if (code === 3) return { sky: 'cloudy', strength: 0 }
  if (code === 45 || code === 48) return { sky: 'fog', strength: 0 }
  if (code >= 51 && code <= 57) return { sky: 'drizzle', strength: code >= 55 ? 0.45 : 0.3 }
  if ((code >= 61 && code <= 67) || (code >= 80 && code <= 82))
    return { sky: 'rain', strength: [61, 66, 80].includes(code) ? 0.5 : [63, 81].includes(code) ? 0.75 : 1 }
  if ((code >= 71 && code <= 77) || code === 85 || code === 86) return { sky: 'snow', strength: [71, 85].includes(code) ? 0.5 : 0.85 }
  return { sky: 'storm', strength: 1 }
}
const SKY_ICON = { clear: Sun, partly: CloudSun, cloudy: Cloud, fog: CloudFog, drizzle: CloudDrizzle, rain: CloudRain, snow: CloudSnow, storm: CloudLightning }
const COMPASS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
interface Content { mesh_tileset: string | null; cloud_tileset: string | null; floor: Floor; ifcs: Array<{ name: string; url: string }> }
interface Work { sid: string; scene: THREE.Scene; mesh: THREE.Group; ifc: THREE.Group | null; ifcLoading: boolean
                 ready: Promise<void>; frame: { lon: number; lat: number; radius: number } | null }

const NEAR_ZOOM = 12                        // the works' models are drawn from this zoom in (the map is flat there)
const PIN_HIDE_ZOOM = 16                    // closer than this, a work whose model stands there shows no pin over it
const FLY_PITCH = 60
const FRAME_SHARE = 0.6                     // the model's width as a share of the view's smaller side, when flown to

/** a glTF file of a tileset's root tile — the work's light model (its address beside the tileset's, its query kept) */
async function rootTileUrl(tilesetUrl: string): Promise<string> {
  const abs = new URL(tilesetUrl, location.href)
  const ts = await fetch(abs.href).then(r => { if (!r.ok) throw new Error(`tileset: HTTP ${r.status}`); return r.json() })
  const uri: string | undefined = ts?.root?.content?.uri ?? ts?.root?.content?.url
  if (!uri) throw new Error('the tileset has no root content')
  const u = new URL(uri, abs)
  for (const [k, v] of abs.searchParams) if (!u.searchParams.has(k)) u.searchParams.set(k, v)
  return u.href
}

/** the work's frame in east-north-up metres at its origin: its heading (the plan's +Y to its azimuth — server
 *  geo.plan_to_enu), glTF's Y up turned to Z up, and (the mesh) the scan's floor adjustment */
function localPlacement(g: Georef, floor: Floor, turnToZUp: boolean): THREE.Matrix4 {
  const m = new THREE.Matrix4().makeRotationZ(-g.heading_deg * Math.PI / 180)
  if (turnToZUp) m.multiply(new THREE.Matrix4().makeRotationX(Math.PI / 2))       // (x, y, z) → (x, -z, y)
  if (floor) {
    const { R, s: k, t } = floor
    m.multiply(new THREE.Matrix4().set(
      k * R[0][0], k * R[0][1], k * R[0][2], t[0],
      k * R[1][0], k * R[1][1], k * R[1][2], t[1],
      k * R[2][0], k * R[2][1], k * R[2][2], t[2],
      0, 0, 0, 1))
  }
  return m
}

/** east-north-up metres at the work's origin → Mapbox's mercator units (y grows southward: north is −y) */
function mercatorOf(g: Georef): THREE.Matrix4 {
  const o = mapboxgl.MercatorCoordinate.fromLngLat([g.lon, g.lat], 0)
  const s = o.meterInMercatorCoordinateUnits()
  return new THREE.Matrix4().makeTranslation(o.x, o.y, o.z).scale(new THREE.Vector3(s, -s, s))
}

/** a loaded model shown as the photographs made it: its textures unlit (they carry their own light), both faces */
function unlit(root: THREE.Object3D): void {
  root.traverse(o => {
    const mesh = o as THREE.Mesh
    if (!mesh.isMesh) return
    const old = mesh.material as THREE.MeshStandardMaterial
    const map = (old as { map?: THREE.Texture | null }).map ?? null
    mesh.material = new THREE.MeshBasicMaterial({ map, color: map ? 0xffffff : old.color, side: THREE.DoubleSide })
    old.dispose()
  })
}

export default function MapView({ projects, focus, ifcOn }: MapViewProps) {
  const t = useT()
  const host = useRef<HTMLDivElement>(null)
  const mapRef = useRef<mapboxgl.Map | null>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const projectsRef = useRef(projects)
  projectsRef.current = projects
  const ifcOnRef = useRef(ifcOn)
  ifcOnRef.current = ifcOn
  // the three.js side, drawn inside Mapbox's frame
  const three = useRef<{ renderer: THREE.WebGLRenderer; camera: THREE.Camera; loader: GLTFLoader } | null>(null)
  const works = useRef(new Map<string, Work>())
  const markers = useRef(new Map<string, mapboxgl.Marker>())
  const framedRef = useRef(false)
  // THE WEATHER AT THE WORK FLOWN TO (USER 2026-11-28: "agregale las condiciones climaticas"): its card, and its rain or
  // snow over the map while the camera is near it
  const [weather, setWeather] = useState<Weather | null>(null)
  const weatherRef = useRef<Weather | null>(null)
  weatherRef.current = weather

  // THE MAP, once
  useEffect(() => {
    let gone = false
    void (async () => {
      try {
        const cfg = await fetch('/api/geo/map_config').then(r => r.json()) as { mapbox_token: string | null }
        if (!cfg.mapbox_token) throw new Error(t('globe.noToken'))
        if (gone || !host.current) return
        mapboxgl.accessToken = cfg.mapbox_token
        const map = new mapboxgl.Map({
          container: host.current,
          style: 'mapbox://styles/mapbox/standard',
          config: { basemap: { lightPreset: 'day', theme: 'default', showPointOfInterestLabels: false,
                               showTransitLabels: false, showPedestrianRoads: false } },
          projection: 'globe', center: [0, 20], zoom: 1.4, maxZoom: 24, antialias: true, attributionControl: true,
        })
        mapRef.current = map
        map.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), 'top-right')
        map.on('style.load', () => {
          if (gone) return
          map.addLayer({
            id: 'stac-works', type: 'custom', renderingMode: '3d',
            onAdd: (_m, gl) => {
              const renderer = new THREE.WebGLRenderer({ canvas: map.getCanvas(), context: gl as WebGL2RenderingContext, antialias: true })
              renderer.autoClear = false
              const ktx = new KTX2Loader().setTranscoderPath(`${import.meta.env.BASE_URL}basis/`).detectSupport(renderer)
              const loader = new GLTFLoader().setKTX2Loader(ktx).setMeshoptDecoder(MeshoptDecoder)
              three.current = { renderer, camera: new THREE.Camera(), loader }
            },
            render: (_gl, matrix) => {
              const th = three.current
              if (!th || map.getZoom() < NEAR_ZOOM) return
              const view = new THREE.Matrix4().fromArray(matrix as unknown as number[])
              for (const w of works.current.values()) {
                const g = projectsRef.current.find(p => p.session_id === w.sid)?.georef
                if (!g || !(w.mesh.children.length || w.ifc?.visible)) continue
                // Mapbox's mercator → clip, after the work's east-north-up metres → mercator; each group's own frame
                // (heading, axes, floor) is its matrix in the scene
                th.camera.projectionMatrix.copy(view).multiply(mercatorOf(g))
                th.camera.projectionMatrixInverse.copy(th.camera.projectionMatrix).invert()
                th.renderer.resetState()
                th.renderer.render(w.scene, th.camera)
              }
            },
          })
          setReady(true)
        })
        map.on('error', e => { if (!gone) console.warn('[map]', e.error) })
        map.on('zoom', () => { pinsByZoom.current(); skyOnMap.current() })
      } catch (err) {
        if (!gone) setError(err instanceof Error ? err.message : String(err))
      }
    })()
    return () => {
      gone = true
      for (const w of works.current.values()) w.scene.traverse(o => {
        const m = o as THREE.Mesh
        if (m.isMesh || (o as THREE.Points).isPoints) { m.geometry.dispose(); (m.material as THREE.Material).dispose() }
      })
      works.current.clear()
      markers.current.clear()
      mapRef.current?.remove()
      mapRef.current = null
      three.current = null
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // A WORK'S LIGHT MODEL, loaded once (its mesh's root tile, else its cloud's points)
  const loadWork = useRef<(sid: string) => void>(() => {})
  loadWork.current = (sid: string) => {
    const th = three.current, map = mapRef.current
    if (!th || !map || works.current.has(sid)) return
    const scene = new THREE.Scene()
    const mesh = new THREE.Group()
    mesh.matrixAutoUpdate = false
    scene.add(mesh)
    let done: () => void = () => {}
    const w: Work = { sid, scene, mesh, ifc: null, ifcLoading: false, frame: null, ready: new Promise<void>(r => { done = r }) }
    works.current.set(sid, w)
    void (async () => {
      try {
        const c = await fetch(`/api/geo/${encodeURIComponent(sid)}/content`, { cache: 'no-store' }).then(r => r.json()) as Content
        const g = projectsRef.current.find(p => p.session_id === sid)?.georef
        if (!g) return
        const url = c.mesh_tileset ?? c.cloud_tileset
        if (!url) { setNotice(t('globe.noContent', { name: sid })); return }
        const gltf = await th.loader.loadAsync(await rootTileUrl(url))
        if (works.current.get(sid) !== w) return
        if (c.mesh_tileset) unlit(gltf.scene)
        else gltf.scene.traverse(o => {                     // the cloud's points, a size on screen
          const p = o as THREE.Points
          if (p.isPoints) (p.material as THREE.PointsMaterial).size = 2
        })
        mesh.matrix.copy(localPlacement(g, c.mesh_tileset ? c.floor : null, true))
        mesh.add(gltf.scene)
        // where it stands and how large it is, in east-north-up metres: what the flight frames
        mesh.updateMatrixWorld(true)
        const box = new THREE.Box3().setFromObject(mesh)
        if (!box.isEmpty()) {
          const c0 = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3())
          const lat = g.lat + c0.y / 110540, lon = g.lon + c0.x / (111320 * Math.cos(g.lat * Math.PI / 180))
          w.frame = { lon, lat, radius: Math.max(2, 0.5 * Math.hypot(size.x, size.y, size.z)) }
        }
        map.triggerRepaint()
        loadIfc.current(sid, c)
      } catch (err) {
        works.current.delete(sid)
        setNotice(t('globe.workFailed', { name: sid, detail: err instanceof Error ? err.message : String(err) }))
      } finally {
        done()
        pinsByZoom.current()
      }
    })()
  }

  // A WORK'S IFC, while ticked in the list: the root tile of the IFC's tiles (bim/ifc_tiles.py)
  const loadIfc = useRef<(sid: string, c?: Content) => void>(() => {})
  loadIfc.current = (sid: string, c?: Content) => {
    const w = works.current.get(sid), th = three.current, map = mapRef.current
    if (!w || !th || !map) return
    const on = ifcOnRef.current.includes(sid)
    if (w.ifc) { w.ifc.visible = on; map.triggerRepaint(); return }
    if (!on || w.ifcLoading) return
    w.ifcLoading = true
    void (async () => {
      try {
        const content = c ?? await fetch(`/api/geo/${encodeURIComponent(sid)}/content`, { cache: 'no-store' }).then(r => r.json()) as Content
        const g = projectsRef.current.find(p => p.session_id === sid)?.georef
        if (!g || !content.ifcs.length) return
        const { fetchIfcTiles } = await import('../../components/ifcTiles')
        const grp = new THREE.Group()
        grp.matrixAutoUpdate = false
        grp.matrix.copy(localPlacement(g, null, true))           // written from the scan: its floor adjustment is in it
        for (const ifc of content.ifcs) {
          const info = await fetchIfcTiles(ifc.url, { onBuilding: () => setNotice(t('bim.tilesBuilding', { name: ifc.name })) })
          const gltf = await th.loader.loadAsync(await rootTileUrl(info.tilesetUrl))
          const colours = new Map(info.parts.map(p => [p.name, p.color]))
          gltf.scene.traverse(o => {
            const mesh = o as THREE.Mesh
            if (!mesh.isMesh) return
            const col = colours.get((mesh.geometry.userData as { name?: string }).name ?? '') ?? [0.7, 0.7, 0.7, 1]
            ;(mesh.material as THREE.Material).dispose()
            mesh.material = new THREE.MeshStandardMaterial({ color: new THREE.Color(col[0], col[1], col[2]), flatShading: true,
              side: THREE.DoubleSide, roughness: 0.7, metalness: 0, transparent: true, opacity: 0.85 })
          })
          grp.add(gltf.scene)
        }
        grp.add(new THREE.HemisphereLight(0xffffff, 0x404040, 2.2), new THREE.DirectionalLight(0xffffff, 1.2))
        if (works.current.get(sid) !== w) return
        w.ifc = grp
        w.scene.add(grp)
        grp.visible = ifcOnRef.current.includes(sid)
        setNotice(null)
        map.triggerRepaint()
      } catch (err) {
        setNotice(t('globe.workFailed', { name: sid, detail: err instanceof Error ? err.message : String(err) }))
      } finally {
        w.ifcLoading = false
      }
    })()
  }

  // A WORK FLOWN TO: its model loaded first (a second: its light model), then ONE flight that frames it whole, tilted
  const flyToWork = useRef<(sid: string) => void>(() => {})
  flyToWork.current = (sid: string) => {
    const map = mapRef.current
    const g = projectsRef.current.find(p => p.session_id === sid)?.georef
    if (!map || !g) return
    loadWork.current(sid)
    fetchWeather(sid)
    const w = works.current.get(sid)
    void (w?.ready ?? Promise.resolve()).then(() => {
      const f = works.current.get(sid)?.frame
      const lat = f?.lat ?? g.lat, lon = f?.lon ?? g.lon, radius = f?.radius ?? 30
      const canvas = map.getCanvas()
      const side = Math.min(canvas.clientWidth, canvas.clientHeight) || 600
      const mpp = (2 * radius) / (FRAME_SHARE * side)                    // metres a pixel that frame it
      const zoom = Math.min(23.5, Math.log2(156543.03392 * Math.cos(lat * Math.PI / 180) / mpp))
      map.flyTo({ center: [lon, lat], zoom, pitch: FLY_PITCH, bearing: map.getBearing() || -20, duration: 4000, essential: true })
    })
  }

  // the weather's rain or snow on the map: near the work it was asked for, as hard as it falls there now
  const skyOnMap = useRef<() => void>(() => {})
  skyOnMap.current = () => {
    const map = mapRef.current, w = weatherRef.current
    if (!map) return
    const near = !!w && map.getZoom() >= NEAR_ZOOM
    const { sky, strength } = w ? skyOf(w.code) : { sky: 'clear' as Sky, strength: 0 }
    const wet = near && (sky === 'rain' || sky === 'drizzle' || sky === 'storm')
    const white = near && sky === 'snow'
    map.setRain(wet ? { density: strength, intensity: strength, color: '#a8adbc', opacity: 0.7, vignette: 0.45,
                        'vignette-color': '#464646', direction: [w!.wind_dir_deg, 75], 'droplet-size': [2.4, 16], 'center-thinning': 0.2 }
                    : undefined)
    map.setSnow(white ? { density: strength, intensity: strength, color: '#ffffff', opacity: 0.9, vignette: 0.3,
                          'vignette-color': '#ffffff', direction: [w!.wind_dir_deg, 50], 'flake-size': 0.7, 'center-thinning': 0.2 }
                      : undefined)
  }
  const fetchWeather = (sid: string) => {
    void fetch(`/api/geo/${encodeURIComponent(sid)}/weather`)
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (d?.ok) { setWeather({ sid, ...d }); weatherRef.current = { sid, ...d } } else setWeather(null); skyOnMap.current() })
      .catch(() => setWeather(null))
  }

  // a work's pin hides when its model stands there and the camera is close (the pin covered it)
  const pinsByZoom = useRef<() => void>(() => {})
  pinsByZoom.current = () => {
    const map = mapRef.current
    if (!map) return
    const near = map.getZoom() >= PIN_HIDE_ZOOM
    for (const [sid, mk] of markers.current) {
      const shown = works.current.get(sid)?.mesh.children.length
      mk.getElement().style.visibility = near && shown ? 'hidden' : 'visible'
    }
  }

  // THE PINS: one per placed work; a touch flies there and stands its model
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map) return
    const seen = new Set<string>()
    for (const p of projects) {
      if (!p.georef) continue
      seen.add(p.session_id)
      let mk = markers.current.get(p.session_id)
      if (!mk) {
        const el = document.createElement('button')
        el.type = 'button'
        el.className = 'stac-map-pin'
        el.title = p.session_id
        const dot = document.createElement('span'); dot.className = 'stac-map-pin__dot'
        const label = document.createElement('span'); label.className = 'stac-map-pin__label'; label.textContent = p.session_id
        el.append(dot, label)
        el.addEventListener('click', ev => { ev.stopPropagation(); flyToWork.current(p.session_id) })
        mk = new mapboxgl.Marker({ element: el, anchor: 'bottom' }).setLngLat([p.georef.lon, p.georef.lat]).addTo(map)
        markers.current.set(p.session_id, mk)
      } else mk.setLngLat([p.georef.lon, p.georef.lat])
    }
    for (const [sid, mk] of markers.current) if (!seen.has(sid)) { mk.remove(); markers.current.delete(sid) }
    // the globe turned once to the works placed (it opened over the Atlantic: a work in America lay behind it)
    const placed = projects.filter(p => p.georef).map(p => p.georef!)
    if (!framedRef.current && placed.length) {
      framedRef.current = true
      const lon = placed.reduce((a, g) => a + g.lon, 0) / placed.length
      const lat = placed.reduce((a, g) => a + g.lat, 0) / placed.length
      map.jumpTo({ center: [lon, lat], zoom: 2.2 })
    }
    // a work placed again: its model to its new place
    for (const w of works.current.values()) if (!projects.some(p => p.session_id === w.sid && p.georef)) works.current.delete(w.sid)
    map.triggerRepaint()
  }, [ready, projects])

  // GO TO A WORK (the list's name)
  useEffect(() => { if (ready && focus) flyToWork.current(focus.sid) }, [ready, focus])

  // ITS IFC, ticked or not in the list
  useEffect(() => {
    if (!ready) return
    for (const sid of ifcOn) { if (!works.current.has(sid)) loadWork.current(sid); else loadIfc.current(sid) }
    for (const w of works.current.values()) if (w.ifc && !ifcOn.includes(w.sid)) w.ifc.visible = false
    mapRef.current?.triggerRepaint()
  }, [ready, ifcOn])

  // a place found by the search: the camera goes there (a region: its whole extent)
  const goTo = (p: Place) => {
    const map = mapRef.current
    if (!map) return
    if (p.extent) map.fitBounds([[p.extent.west, p.extent.south], [p.extent.east, p.extent.north]], { duration: 2500 })
    else map.flyTo({ center: [p.lon, p.lat], zoom: 16, duration: 2500 })
  }

  return (
    <div className="stac-globe" aria-label={t('globe.title')}>
      <div ref={host} className="stac-globe__canvas" />
      {ready && <div className="stac-globe__search"><PlaceSearch onPick={goTo} /></div>}
      {!ready && !error && <div className="stac-globe__status">{t('globe.loading')}</div>}
      {error && <div className="stac-globe__status stac-globe__status--error" role="alert">{t('globe.failed', { detail: error })}</div>}
      {notice && <button type="button" className="stac-globe__notice" onClick={() => setNotice(null)}>{notice}</button>}
      {weather && (() => {
        const { sky } = skyOf(weather.code)
        const Icon = SKY_ICON[sky]
        return (
          <div className="stac-map-weather" role="status" aria-label={t('globe.weatherTitle', { name: weather.sid })}>
            <div className="stac-map-weather__main">
              <Icon aria-hidden className="stac-map-weather__icon" />
              <div>
                <div className="stac-map-weather__temp">{Math.round(weather.temperature_c)}°C</div>
                <div className="stac-map-weather__sky">{t(`globe.sky_${sky}`)}</div>
              </div>
            </div>
            <div className="stac-map-weather__rows">
              <span>{t('globe.feelsLike', { t: Math.round(weather.apparent_c) })}</span>
              <span>{t('globe.humidity', { h: Math.round(weather.humidity_pct) })}</span>
              <span>{t('globe.wind', { v: Math.round(weather.wind_kmh), dir: COMPASS[Math.round(weather.wind_dir_deg / 45) % 8] })}</span>
              {weather.precip_mm > 0 && <span>{t('globe.precip', { mm: weather.precip_mm.toFixed(1) })}</span>}
            </div>
            <div className="stac-map-weather__foot">{t('globe.weatherNow', { name: weather.sid })} · Open-Meteo</div>
          </div>
        )
      })()}
    </div>
  )
}
