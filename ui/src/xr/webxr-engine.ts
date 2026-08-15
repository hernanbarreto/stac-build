// STAC-Builder — WebXR engine (ARKit-grade tracking) for the mobile viewer.
//
// Runs wherever real WebXR immersive-ar exists: Android Chrome natively, and
// iPhone through the Variant Launch App Clip (which exposes Apple's ARKit to
// the page — rock-solid anchoring, true metric scale, hit-test on REAL
// surfaces; a league above any in-browser JS SLAM).
//
// dom-overlay keeps the React UI (toolbar/topbar/toast) alive during the
// session, so tools are switched with the normal buttons; screen taps arrive
// as XR 'select' events and are resolved through the input ray.
//
// Hernán Barreto - Ingerop IN3 Session IV - STAC

import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js'
import { MeasureKit } from './measure'
import { SCALES, tele, type EngineCallbacks, type IXREngine, type Tool } from './engine-types'

export class WebXREngine implements IXREngine {
  private renderer: THREE.WebGLRenderer | null = null
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(60, 1, 0.02, 300)
  private modelGroup = new THREE.Group()
  private contentGroup = new THREE.Group()
  private measureGroup = new THREE.Group()
  private measure = new MeasureKit(this.measureGroup)
  private reticle: THREE.Mesh | null = null
  private raycaster = new THREE.Raycaster()
  private raycastTargets: THREE.Object3D[] = []
  private lastAnchor: THREE.Vector3 | null = null
  private hitTestSource: any = null
  private refSpace: XRReferenceSpace | null = null
  private session: XRSession | null = null

  placed = false
  tool: Tool = 'move'
  scaleIdx = 0

  constructor(private sessionId: string, private cb: EngineCallbacks) {
    ;(this.raycaster as any).firstHitOnly = true
  }

  async start(canvas: HTMLCanvasElement) {
    try {
      this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
      this.renderer.setClearColor(0x000000, 0)
      this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
      this.renderer.setSize(innerWidth, innerHeight)
      this.renderer.xr.enabled = true

      this.scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.0))
      this.modelGroup.add(this.contentGroup, this.measureGroup)
      this.contentGroup.visible = false           // nothing until placed
      this.scene.add(this.modelGroup)

      this.reticle = new THREE.Mesh(
        new THREE.RingGeometry(0.07, 0.09, 32).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: 0x3fb950, depthTest: false }))
      this.reticle.renderOrder = 998
      this.reticle.matrixAutoUpdate = false
      this.reticle.visible = false
      this.scene.add(this.reticle)

      this.setScaleIdx(0)
      // Session FIRST, inside the tap gesture (iOS requires it) — the camera
      // shows immediately; the mesh streams in the background and announces
      // itself when ready. Awaiting the 33 MB download first both blanked the
      // screen AND dropped the user-gesture context for requestSession.
      this.loadMesh()
        .then(() => this.cb.onToast('Mesh loaded — aim the circle and tap to place'))
        .catch((e) => {
          tele('mesh-error', { msg: String(e?.message ?? e) })
          this.cb.onError(`Mesh load failed: ${e?.message ?? e}`)
        })

      const overlayRoot = document.getElementById('root') ?? document.body
      const session = await (navigator as any).xr.requestSession('immersive-ar', {
        requiredFeatures: [],
        optionalFeatures: ['local-floor', 'hit-test', 'dom-overlay'],
        domOverlay: { root: overlayRoot },
      })
      this.session = session
      let refType: XRReferenceSpaceType = 'local-floor'
      try { await session.requestReferenceSpace('local-floor') }
      catch { refType = 'local' }
      this.renderer.xr.setReferenceSpaceType(refType)
      await this.renderer.xr.setSession(session)
      this.refSpace = this.renderer.xr.getReferenceSpace()
      try {
        const viewer = await session.requestReferenceSpace('viewer')
        this.hitTestSource = await (session as any)
          .requestHitTestSource?.({ space: viewer }) ?? null
      } catch { this.hitTestSource = null }

      // page background transparent while the session runs — dom-overlay
      // draws the page OVER the camera; opaque dark bg = no passthrough
      document.documentElement.classList.add('xr-session')
      tele('webxr-start', {
        session: this.sessionId, refType,
        blend: (session as any).environmentBlendMode,
        hitTest: !!this.hitTestSource,
        domOverlay: !!(session as any).domOverlayState,
      })
      session.addEventListener('select', (ev: any) => this.onSelect(ev))
      session.addEventListener('end', () => {
        document.documentElement.classList.remove('xr-session')
        this.cb.onError('AR session ended')
      })
      this.renderer.setAnimationLoop((_t, frame) => this.onFrame(frame))
      this.cb.onTracking('NORMAL')                // ARKit is solid by contract
      this.cb.onReady()
    } catch (e: any) {
      tele('webxr-error', { msg: String(e?.message ?? e).slice(0, 300) })
      this.cb.onError(`AR start failed: ${e?.message ?? e}`)
    }
  }

  stop() {
    try { this.session?.end() } catch { /* already gone */ }
    this.renderer?.setAnimationLoop(null)
    this.renderer?.dispose()
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
        o.material = new THREE.MeshBasicMaterial({
          map: o.material?.map ?? null,
          color: o.material?.color?.clone() ?? new THREE.Color(0xffffff),
        })
      }
    })
    this.contentGroup.add(gltf.scene)
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

  private onFrame(frame: any) {
    if (frame && this.hitTestSource && this.reticle) {
      if (this.tool === 'move') {
        const hits = frame.getHitTestResults(this.hitTestSource)
        let shown = false
        for (const h of hits) {
          const pose = h.getPose(this.refSpace!)
          // FLOOR-ONLY placement: ARKit hit-tests walls too — anchoring the
          // model floor at a wall point floated it mid-air. Accept only
          // horizontal-up surfaces (hit pose +Y ≈ world up).
          const m = new THREE.Matrix4().fromArray(pose.transform.matrix)
          const up = new THREE.Vector3(0, 1, 0)
            .applyMatrix4(new THREE.Matrix4().extractRotation(m))
          if (up.y > 0.85) {
            this.reticle.matrix.copy(m)
            this.reticle.visible = true
            shown = true
            break
          }
        }
        if (!shown) this.reticle.visible = false
      } else {
        this.reticle.visible = false
      }
    }
    this.renderer!.render(this.scene, this.camera)
  }

  private onSelect(ev: any) {
    const frame = ev.frame
    if (this.tool === 'move') {
      if (this.reticle?.visible) {
        const p = new THREE.Vector3(), q = new THREE.Quaternion(),
              s = new THREE.Vector3()
        this.reticle.matrix.decompose(p, q, s)
        // ARKit hit-test lands on REAL surfaces — trust its height directly
        this.placeAt(p.x, p.y, p.z)
        this.cb.onPlaced()
        tele('webxr-placed', { scale: SCALES[this.scaleIdx][1] })
      }
      return
    }
    if (!this.placed) { this.cb.onToast('Place the model first (Move + tap)'); return }
    const pose = frame?.getPose(ev.inputSource.targetRaySpace, this.refSpace)
    if (!pose) return
    const m = new THREE.Matrix4().fromArray(pose.transform.matrix)
    const origin = new THREE.Vector3().setFromMatrixPosition(m)
    const dir = new THREE.Vector3(0, 0, -1)
      .applyMatrix4(new THREE.Matrix4().extractRotation(m)).normalize()
    this.raycaster.set(origin, dir)
    ;(this.raycaster as any).camera = this.renderer!.xr.getCamera()
    const hits = this.raycaster.intersectObjects(this.raycastTargets, false)
    if (!hits.length) { this.cb.onToast('No mesh under the tap'); return }
    this.measure.addPoint(this.tool,
      this.modelGroup.worldToLocal(hits[0].point.clone()))
  }

  private placeAt(x: number, y: number, z: number) {
    this.contentGroup.visible = true
    this.placed = true
    this.modelGroup.updateWorldMatrix(true, true)
    const box = new THREE.Box3().setFromObject(this.contentGroup)
    if (box.isEmpty()) return
    const c = box.getCenter(new THREE.Vector3())
    this.modelGroup.position.x += x - c.x
    this.modelGroup.position.z += z - c.z
    this.modelGroup.position.y = y              // model floor on the REAL surface
    this.lastAnchor = new THREE.Vector3(x, y, z)
    // ground-truth numbers for remote diagnosis: where did it actually land
    this.modelGroup.updateWorldMatrix(true, true)
    const after = new THREE.Box3().setFromObject(this.contentGroup)
    const cam = this.renderer!.xr.getCamera()
    tele('place-geom', {
      hitY: +y.toFixed(2),
      boxMinY: +after.min.y.toFixed(2), boxMaxY: +after.max.y.toFixed(2),
      boxMinX: +after.min.x.toFixed(1), boxMaxX: +after.max.x.toFixed(1),
      camY: +cam.position.y.toFixed(2),
      scale: SCALES[this.scaleIdx][1],
    })
  }

  setTool(tool: Tool) {
    this.tool = tool
    this.measure.resetPending()
  }

  setScaleIdx(i: number): string {
    this.scaleIdx = i % SCALES.length
    this.modelGroup.scale.setScalar(SCALES[this.scaleIdx][0])
    if (this.lastAnchor) {
      this.placeAt(this.lastAnchor.x, this.lastAnchor.y, this.lastAnchor.z)
    }
    return SCALES[this.scaleIdx][1]
  }

  recenter() { /* ARKit tracking needs no manual recenter */ }

  clearMeasures() { this.measure.clear() }

  tap(_x: number, _y: number): string | null {
    // immersive taps arrive as XR 'select' events, not DOM touches
    return null
  }
}
