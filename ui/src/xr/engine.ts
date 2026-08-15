// STAC-Builder — XR engine layer for the mobile viewer (React-free core).
//
// Wraps the self-hosted 8th Wall engine (window.XR8): SLAM world tracking with
// ABSOLUTE metric scale over the live camera feed — real AR in iOS Safari
// (which has no WebXR in 2026) and Android browsers alike, no accounts, no
// external services. three.js rides the XR8.Threejs camera pipeline; the STAC
// mesh loads with the meshopt decoder and BVH-accelerated raycasting so
// measurement taps answer in milliseconds on multi-M-triangle scenes.
//
// All measurement math happens in METRIC MODEL SPACE (the reconstruction's
// meters), so values stay true at any display scale (1:10 tabletop or 1:1).
//
// Hernán Barreto - Ingerop IN3 Session IV - STAC

import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js'
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh'

;(THREE.BufferGeometry.prototype as any).computeBoundsTree = computeBoundsTree
;(THREE.BufferGeometry.prototype as any).disposeBoundsTree = disposeBoundsTree
;(THREE.Mesh.prototype as any).raycast = acceleratedRaycast

// The 8th Wall ThreeJS pipeline module requires a GLOBAL three (it does not
// bundle its own): without this line XR8.Threejs.pipelineModule() throws and
// every AR start dies before the camera — the root cause of the silent stalls.
;(window as any).THREE = THREE

declare global {
  interface Window { XR8: any }
}

export type Tool = 'move' | 'dist' | 'angle' | 'vol'
// METRIC FIRST (user mandate): 1:1 is the default; miniatures are the option
export const SCALES: Array<[number, string]> = [[1, '1:1'], [0.1, '1:10'], [0.02, '1:50']]

export interface EngineCallbacks {
  onReady: () => void
  onError: (msg: string) => void
  onToast: (msg: string) => void
  onPlaced: () => void
  onTracking: (status: string) => void
}

export function tele(event: string, data: Record<string, unknown> = {}) {
  try {
    fetch('/api/ar/log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event: `xr-${event}`, ...data }),
    }).catch(() => {})
  } catch { /* telemetry must never break the app */ }
}

// XR browsers have no devtools: EVERY uncaught exception must reach the pod
// log, or failures die silently (which is exactly what burned this whole day)
window.addEventListener('error', (e) =>
  tele('js-error', { msg: String(e.message), src: `${e.filename}:${e.lineno}` }))
window.addEventListener('unhandledrejection', (e) =>
  tele('promise-rejection', { msg: String((e as any).reason).slice(0, 300) }))

export class StacXREngine {
  private scene: THREE.Scene | null = null
  private camera: THREE.PerspectiveCamera | null = null
  private modelGroup = new THREE.Group()
  private contentGroup = new THREE.Group()
  private measureGroup = new THREE.Group()
  private reticle: THREE.Mesh | null = null
  private raycaster = new THREE.Raycaster()
  private raycastTargets: THREE.Object3D[] = []
  private lastAnchor: THREE.Vector3 | null = null
  private pending: THREE.Vector3[] = []
  private running = false

  placed = false
  tool: Tool = 'move'
  scaleIdx = 0
  trackingStatus = 'UNSPECIFIED'   // SLAM needs device TRANSLATION to reach
                                   // NORMAL — until then content slides and
                                   // scale is not metric (root of both bugs)
  private lastHitTele = 0
  // MEASURED floor height: the engine's theoretical ground (world y=0) assumes
  // the phone started exactly at the configured eye height — holding it lower/
  // higher floats or sinks the model. Estimate the real floor from hits that
  // sit at plausible floor depth below the camera (0.8–2.5 m), smoothed.
  private floorY = 0
  private floorSamples = 0

  constructor(
    private sessionId: string,
    private cb: EngineCallbacks,
  ) {
    ;(this.raycaster as any).firstHitOnly = true
  }

  start(canvas: HTMLCanvasElement) {
    const boot = () => this.startEngine(canvas)
    if (window.XR8) boot()
    else window.addEventListener('xrloaded', boot, { once: true })
  }

  stop() {
    this.running = false
    try { window.XR8?.stop() } catch { /* engine may not have started */ }
  }

  private cameraStarted = false

  private startEngine(canvas: HTMLCanvasElement) {
    const XR8 = window.XR8
    this.running = true
    tele('engine-loaded', { version: XR8.version ?? '?' })
    // FULL-WINDOW CANVAS: the engine renders at the canvas's pixel size — the
    // default 300×150 showed the camera as a tiny corner rectangle. (XRExtras'
    // FullWindowCanvas module does this in stock 8th Wall setups.)
    // FROZEN canvas size for the whole AR session: ANY canvas.width/height
    // reassignment resets the GL context → the engine restarts the camera →
    // the SLAM loses its map (the drift the user saw). iOS resize events fire
    // constantly as Safari's bars animate — so we size ONCE and never again.
    canvas.width = document.documentElement.clientWidth
    canvas.height = document.documentElement.clientHeight
    // FAIL LOUD on unsupported browsers: on iPhone every browser is Safari's
    // engine by Apple mandate — Chrome-iOS stalls silently without this check
    try {
      const compat = XR8.XrDevice?.isDeviceBrowserCompatible?.()
      if (compat === false) {
        const reasons = XR8.XrDevice?.incompatibleReasons?.() ?? []
        tele('incompatible', { reasons })
        const ios = /iPhone|iPad|iPod/.test(navigator.userAgent)
        this.cb.onError(ios
          ? 'This browser cannot run AR on iPhone. Apple forces every iOS '
            + 'browser (Chrome included) onto Safari\'s engine — open this '
            + 'page in Safari, the only one the AR engine supports there.'
          : 'This browser cannot run AR on this device — use Chrome on Android.')
        return
      }
    } catch { /* older engine builds may lack the API */ }
    // watchdog: a camera that never reports within 12 s is a silent stall
    setTimeout(() => {
      if (this.running && !this.cameraStarted) {
        tele('camera-stall', {})
        const ios = /iPhone|iPad|iPod/.test(navigator.userAgent)
        this.cb.onError(ios
          ? 'The camera never started. On iPhone, open this page in Safari '
            + '(Apple blocks the AR engine in every other browser).'
          : 'The camera never started — check the camera permission for this '
            + 'site and reload.')
      }
    }, 12000)
    try {
      XR8.XrController.configure({ scale: 'absolute' })   // METRIC world units
      XR8.addCameraPipelineModules([
        XR8.GlTextureRenderer.pipelineModule(),           // camera feed
        XR8.Threejs.pipelineModule(),                     // three scene from SLAM
        XR8.XrController.pipelineModule(),                // 6DoF tracking
        this.pipelineModule(),
      ])
      XR8.run({ canvas })
      tele('run-called', {})
    } catch (e: any) {
      tele('start-exception', { msg: String(e?.message ?? e).slice(0, 300),
                                stack: String(e?.stack ?? '').slice(0, 300) })
      this.cb.onError(`Engine start failed: ${e?.message ?? e}`)
    }
  }

  private pipelineModule() {
    return {
      name: 'stac-xr',
      onStart: () => {
        const s = window.XR8.Threejs.xrScene()
        this.scene = s.scene
        this.camera = s.camera
        this.camera!.position.set(0, 1.6, 0)            // standing eye height

        this.scene!.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.0))
        this.modelGroup.add(this.contentGroup, this.measureGroup)
        this.contentGroup.visible = false               // nothing until placed
        this.scene!.add(this.modelGroup)

        this.reticle = new THREE.Mesh(
          new THREE.RingGeometry(0.07, 0.09, 32).rotateX(-Math.PI / 2),
          new THREE.MeshBasicMaterial({ color: 0x3fb950, depthTest: false }))
        this.reticle.renderOrder = 998
        this.reticle.visible = false
        this.scene!.add(this.reticle)

        this.setScaleIdx(0)
        this.loadMesh()
          .then(() => { this.cb.onReady(); tele('ready', { session: this.sessionId }) })
          .catch((e) => {
            this.cb.onError(`Mesh load failed: ${e?.message ?? e}`)
            tele('mesh-error', { msg: String(e?.message ?? e) })
          })
      },
      onUpdate: (args: any) => {
        // SLAM tracking status: LIMITED = sliding content + non-metric scale.
        // Surface it so the UI can coach the user into initializing (walk a
        // step) and so placement is gated on NORMAL.
        const reality = args?.processCpuResult?.reality
        const ts = reality?.trackingStatus
        if (ts && ts !== this.trackingStatus) {
          this.trackingStatus = ts
          tele('tracking', { status: ts, reason: reality?.trackingReason })
          this.cb.onTracking(ts)
        }
        if (!this.scene || !this.reticle) return
        if (this.tool !== 'move' || this.trackingStatus !== 'NORMAL') {
          this.reticle.visible = false
          return
        }
        // surfaces preferred; feature points as FALLBACK (less precise, but a
        // reticle that never appears is worse). Hit counts go to telemetry
        // once per second so the surface-detection health is visible remotely.
        const surf = window.XR8.XrController.hitTest(0.5, 0.5,
          ['DETECTED_SURFACE', 'ESTIMATED_SURFACE'])
        const feat = surf?.length ? [] : window.XR8.XrController.hitTest(
          0.5, 0.5, ['FEATURE_POINT'])
        const now = Date.now()
        if (now - this.lastHitTele > 1000) {
          this.lastHitTele = now
          tele('hit-test', { surf: surf?.length ?? 0, feat: feat?.length ?? 0,
                             tracking: this.trackingStatus })
        }
        const h = (surf && surf[0]) || (feat && feat[0])
        if (h) {
          // floor-like hits (well below the camera) refine the measured floor
          const camY = this.camera!.position.y
          if (h.position.y < camY - 0.8 && h.position.y > camY - 2.5) {
            this.floorY = this.floorSamples === 0
              ? h.position.y
              : this.floorY * 0.8 + h.position.y * 0.2
            this.floorSamples++
          }
          this.reticle.position.set(h.position.x, this.floorY, h.position.z)
          const d = Math.max(
            this.reticle.position.distanceTo(this.camera!.position) / 6, 0.6)
          this.reticle.scale.setScalar(d)
          this.reticle.visible = true
        } else {
          this.reticle.visible = false
        }
      },
      onCameraStatusChange: (e: any) => {
        tele('camera-status', { status: e?.status })
        if (e?.status && e.status !== 'failed') this.cameraStarted = true
        if (e?.status === 'failed') {
          this.cb.onError('Camera access failed — allow the camera for this '
            + 'site (aA menu → Website settings) and retry')
        }
      },
      listeners: [{
        event: 'reality.error',
        process: (e: any) => {
          tele('reality-error', { msg: String(e?.detail ?? e) })
          this.cb.onError('Camera/tracking failed — allow camera + motion access and reload')
        },
      }],
    }
  }

  private async loadMesh() {
    const loader = new GLTFLoader()
    loader.setMeshoptDecoder(MeshoptDecoder)
    const gltf = await loader.loadAsync(
      `/api/ar/mesh/${encodeURIComponent(this.sessionId)}`)
    gltf.scene.traverse((o: any) => {
      if (o.isMesh) {
        o.geometry.computeBoundsTree()
        this.raycastTargets.push(o)
        // fullbright: photogrammetry textures already carry the lighting
        o.material = new THREE.MeshBasicMaterial({
          map: o.material?.map ?? null,
          color: o.material?.color?.clone() ?? new THREE.Color(0xffffff),
        })
      }
    })
    this.contentGroup.add(gltf.scene)
    // upright transform for sessions whose in-pipeline orientation was refused
    try {
      const r = await fetch('/api/ar/sessions')
      const d = await r.json()
      const s = d.sessions?.find((x: any) => x.id === this.sessionId)
      if (s?.floor_transform) {
        this.contentGroup.matrixAutoUpdate = false
        this.contentGroup.matrix.set(...(s.floor_transform as
          [number, number, number, number, number, number, number, number,
           number, number, number, number, number, number, number, number]))
        this.contentGroup.matrixWorldNeedsUpdate = true
      }
    } catch { /* orientation stays as baked */ }
  }

  setTool(tool: Tool) {
    this.tool = tool
    this.pending = []
  }

  setScaleIdx(i: number): string {
    this.scaleIdx = i % SCALES.length
    this.modelGroup.scale.setScalar(SCALES[this.scaleIdx][0])
    if (this.lastAnchor) {
      this.placeAt(this.lastAnchor.x, this.lastAnchor.y, this.lastAnchor.z)
    }
    return SCALES[this.scaleIdx][1]
  }

  recenter() { window.XR8?.XrController.recenter() }

  clearMeasures() {
    this.pending = []
    while (this.measureGroup.children.length) this.measureGroup.children.pop()
  }

  /** The scene origin is wherever the capture walk started — often tens of
   *  meters from the geometry. Place the CONTENT's bbox center on the target,
   *  with the pipeline-calibrated floor plane (model y=0) on the hit height. */
  private placeAt(x: number, _y: number, z: number) {
    this.contentGroup.visible = true
    this.placed = true
    const box = new THREE.Box3().setFromObject(this.contentGroup)
    if (box.isEmpty()) return
    const c = box.getCenter(new THREE.Vector3())
    this.modelGroup.position.x += x - c.x
    this.modelGroup.position.z += z - c.z
    // model floor (pipeline-calibrated y=0) on the MEASURED floor height
    this.modelGroup.position.y = this.floorY
    this.lastAnchor = new THREE.Vector3(x, this.floorY, z)
  }

  /** Single tap from the React layer. Returns a short status for the UI. */
  tap(clientX: number, clientY: number): string | null {
    if (!this.scene || !this.camera) return null
    if (this.tool === 'move') {
      if (this.trackingStatus !== 'NORMAL') return 'tracking'
      if (this.reticle?.visible) {
        const p = this.reticle.position
        this.placeAt(p.x, p.y, p.z)
        this.cb.onPlaced()
        tele('placed', { scale: SCALES[this.scaleIdx][1] })
        return 'placed'
      }
      return 'no-surface'
    }
    if (!this.placed) return 'place-first'
    const ndc = new THREE.Vector2(
      (clientX / window.innerWidth) * 2 - 1,
      -(clientY / window.innerHeight) * 2 + 1)
    this.raycaster.setFromCamera(ndc, this.camera)
    const hits = this.raycaster.intersectObjects(this.raycastTargets, false)
    if (!hits.length) return 'no-mesh'
    this.addMeasurePoint(this.modelGroup.worldToLocal(hits[0].point.clone()))
    return 'measured'
  }

  // ── measurement drawing (metric model space) ──────────────────────────────

  private fmt(m: number) {
    return m >= 1 ? `${m.toFixed(2)} m` : `${(m * 100).toFixed(1)} cm`
  }

  private marker(p: THREE.Vector3, color = 0xf0a839) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 12),
                             new THREE.MeshBasicMaterial({ color }))
    s.position.copy(p)
    this.measureGroup.add(s)
  }

  private line(points: THREE.Vector3[], color = 0xf0a839) {
    this.measureGroup.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color })))
  }

  private label(text: string, p: THREE.Vector3) {
    const pad = 8, fs = 34
    const cv = document.createElement('canvas')
    const cx = cv.getContext('2d')!
    cx.font = `600 ${fs}px system-ui`
    cv.width = cx.measureText(text).width + pad * 2
    cv.height = fs + pad * 2
    cx.font = `600 ${fs}px system-ui`
    cx.fillStyle = 'rgba(10,13,18,0.85)'
    cx.fillRect(0, 0, cv.width, cv.height)
    cx.fillStyle = '#f0a839'
    cx.textBaseline = 'middle'
    cx.fillText(text, pad, cv.height / 2)
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({
      map: new THREE.CanvasTexture(cv), depthTest: false }))
    sp.renderOrder = 999
    const h = 0.12
    sp.scale.set(h * cv.width / cv.height, h, 1)
    sp.position.copy(p)
    this.measureGroup.add(sp)
  }

  private addMeasurePoint(p: THREE.Vector3) {
    if (this.tool === 'dist') {
      this.pending.push(p); this.marker(p)
      if (this.pending.length === 2) {
        const [a, b] = this.pending
        this.line([a, b])
        this.label(this.fmt(a.distanceTo(b)),
                   a.clone().add(b).multiplyScalar(0.5))
        this.pending = []
      }
    } else if (this.tool === 'angle') {
      this.pending.push(p); this.marker(p, 0x4fd1ff)
      if (this.pending.length === 3) {
        const [a, v, b] = this.pending
        this.line([a, v, b], 0x4fd1ff)
        const u1 = a.clone().sub(v).normalize()
        const u2 = b.clone().sub(v).normalize()
        const deg = THREE.MathUtils.radToDeg(
          Math.acos(THREE.MathUtils.clamp(u1.dot(u2), -1, 1)))
        this.label(`${deg.toFixed(1)}°`,
                   v.clone().addScaledVector(u1.add(u2).normalize(), 0.25))
        this.pending = []
      }
    } else if (this.tool === 'vol') {
      this.pending.push(p); this.marker(p, 0x3fb950)
      if (this.pending.length === 3) {
        const [a, b, c] = this.pending
        const y0 = Math.min(a.y, b.y)
        const h = Math.max(Math.abs(c.y - y0), 0.05)
        const min = new THREE.Vector3(Math.min(a.x, b.x), y0, Math.min(a.z, b.z))
        const size = new THREE.Vector3(Math.max(a.x, b.x), y0 + h,
                                       Math.max(a.z, b.z)).sub(min)
        const vol = size.x * size.y * size.z
        const geo = new THREE.BoxGeometry(size.x, size.y, size.z)
        const box = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
          color: 0x3fb950, transparent: true, opacity: 0.18, depthWrite: false }))
        box.position.copy(min).add(size.clone().multiplyScalar(0.5))
        this.measureGroup.add(box)
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo),
          new THREE.LineBasicMaterial({ color: 0x3fb950 }))
        edges.position.copy(box.position)
        this.measureGroup.add(edges)
        this.label(
          `${size.x.toFixed(2)}×${size.z.toFixed(2)}×${size.y.toFixed(2)} m = `
          + `${vol.toFixed(2)} m³`,
          box.position.clone().setY(min.y + size.y + 0.1))
        this.pending = []
      }
    }
  }
}
