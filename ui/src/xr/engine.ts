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

declare global {
  interface Window { XR8: any }
}

export type Tool = 'move' | 'dist' | 'angle' | 'vol'
export const SCALES: Array<[number, string]> = [[0.1, '1:10'], [1, '1:1'], [0.02, '1:50']]

export interface EngineCallbacks {
  onReady: () => void
  onError: (msg: string) => void
  onToast: (msg: string) => void
  onPlaced: () => void
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

  private startEngine(canvas: HTMLCanvasElement) {
    const XR8 = window.XR8
    this.running = true
    tele('engine-loaded', { version: XR8.version ?? '?' })
    XR8.XrController.configure({ scale: 'absolute' })   // METRIC world units
    XR8.addCameraPipelineModules([
      XR8.GlTextureRenderer.pipelineModule(),           // camera feed
      XR8.Threejs.pipelineModule(),                     // three scene from SLAM
      XR8.XrController.pipelineModule(),                // 6DoF tracking
      this.pipelineModule(),
    ])
    XR8.run({ canvas })
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
      onUpdate: () => {
        if (!this.scene || !this.reticle) return
        if (this.tool !== 'move') { this.reticle.visible = false; return }
        const hits = window.XR8.XrController.hitTest(0.5, 0.5,
          ['DETECTED_SURFACE', 'ESTIMATED_SURFACE', 'FEATURE_POINT'])
        const h = hits && hits[0]
        if (h) {
          this.reticle.position.set(h.position.x, h.position.y, h.position.z)
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
  private placeAt(x: number, y: number, z: number) {
    this.contentGroup.visible = true
    this.placed = true
    const box = new THREE.Box3().setFromObject(this.contentGroup)
    if (box.isEmpty()) return
    const c = box.getCenter(new THREE.Vector3())
    this.modelGroup.position.x += x - c.x
    this.modelGroup.position.z += z - c.z
    this.modelGroup.position.y = y
    this.lastAnchor = new THREE.Vector3(x, y, z)
  }

  /** Single tap from the React layer. Returns a short status for the UI. */
  tap(clientX: number, clientY: number): string | null {
    if (!this.scene || !this.camera) return null
    if (this.tool === 'move') {
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
