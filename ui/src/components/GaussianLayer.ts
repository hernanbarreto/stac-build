/**
 * GaussianLayer — the Gaussian-splat realism layer over the metric cloud / mesh (USER 2026-10-31:
 * "el realismo de los GS con la capacidad de medir sobre malla matemáticamente buena como la nuestra …
 * sobre el modelo GS debe poder seleccionarse los objetos tal como lo hacemos actualmente").
 *
 * Renderer: Spark (@sparkjsdev/spark, MIT) inside the existing three.js scene — one SparkRenderer per
 * scene, one SplatMesh per published model, parented under the Potree octree group so the display
 * transform (floor) applies to the splats exactly as to the cloud and the meshes. The splats are what
 * is seen; measurements keep landing on the cloud / mesh (a SplatMesh is not raycastable here).
 *
 * Instances: every splat carries its seed's row in the published cloud; the server resolves the
 * CURRENT instance of each splat live (/api/gaussians/instances — the Segmentation Manager's edits
 * apply with no retraining). Per-splat control travels in an RgbaArray read by a dyno modifier in the
 * splat shader: alpha = shown / hidden, red = highlighted. Hiding an instance in the Instances list
 * hides its splats; selecting one tints them.
 */
import * as THREE from 'three'
import { RgbaArray, SparkRenderer, SplatMesh, dyno, readRgbaArray } from '@sparkjsdev/spark'

export interface GaussianModelInfo {
  name: string
  files: Record<string, string>
  instances_url: string
  n_gaussians: number | null
  stale: boolean
  reason: string
  heldout: { psnr_mean?: number; ssim_mean?: number; n_heldout?: number } | null
  geometry_epoch: number | null
}

interface Entry {
  info: GaussianModelInfo
  mesh: SplatMesh
  control: RgbaArray
  instances: Uint32Array | null
  stale: boolean
  visible: boolean
}

const HIGHLIGHT = new THREE.Vector3(1.0, 0.82, 0.25)      // the selection tint (mixed 60 % into the splat colour)

export class GaussianLayer {
  private spark: SparkRenderer | null = null
  private readonly entries = new Map<string, Entry>()
  private readonly hidden = new Set<number>()
  private highlighted: number | null = null

  constructor(private readonly scene: THREE.Scene, private readonly renderer: THREE.WebGLRenderer) {}

  get names(): string[] { return [...this.entries.keys()] }
  get size(): number { return this.entries.size }

  private ensureRenderer() {
    if (this.spark) return
    this.spark = new SparkRenderer({ renderer: this.renderer })
    this.spark.name = 'spark-renderer'
    this.scene.add(this.spark)
  }

  /** The per-splat control texture (alpha shown / hidden, red highlighted) as the modifier reads it. */
  private makeModifier(control: RgbaArray) {
    const tint = dyno.dynoConst('vec3', HIGHLIGHT)
    const strength = dyno.dynoConst('float', 0.6)
    return dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
      if (!gsplat) return {}
      const s = dyno.splitGsplat(gsplat).outputs
      const c = dyno.split(readRgbaArray(control.dyno, s.index)).outputs      // the control RGBA in [0, 1]
      const opacity = dyno.mul(s.opacity, c.w) as unknown as dyno.DynoVal<'float'>
      const amount = dyno.mul(c.x, strength) as unknown as dyno.DynoVal<'float'>
      const rgb = dyno.mix(s.rgb, tint, amount) as unknown as dyno.DynoVal<'vec3'>
      return { gsplat: dyno.combineGsplat({ gsplat, rgb, opacity }) }
    })
  }

  /** Download a model and add it under `parent`; resolves with the splat count. */
  async load(info: GaussianModelInfo, parent: THREE.Object3D): Promise<number> {
    this.ensureRenderer()
    this.unload(info.name)
    // the 3DGS PLY keeps the view-dependent colour but weighs ~250 B per gaussian (6.5 M = 1.6 GB): over 2 M the
    // compact .splat (32 B each, SH0 only) loads instead — the viewer stays usable; the PLY is the record
    const url = (info.n_gaussians ?? 0) > 2_000_000 ? (info.files.splat ?? info.files.ply) : (info.files.ply ?? info.files.splat)
    if (!url) throw new Error(`gaussians ${info.name}: no .splat or .ply file published`)
    const mesh = new SplatMesh({ url })
    mesh.name = `gaussians-${info.name}`
    mesh.userData = { kind: 'gaussians', name: info.name }
    parent.add(mesh)
    await mesh.initialized
    const n = mesh.numSplats
    const control = new RgbaArray({ capacity: n, count: n })
    control.ensureCapacity(n)
    const entry: Entry = { info, mesh, control, instances: null, stale: info.stale, visible: true }
    this.entries.set(info.name, entry)
    mesh.objectModifier = this.makeModifier(control)
    await this.fetchInstances(entry)
    this.applyControl(entry)
    return n
  }

  private async fetchInstances(entry: Entry): Promise<void> {
    try {
      const r = await fetch(entry.info.instances_url)
      if (!r.ok) { entry.instances = null; return }
      const buf = await r.arrayBuffer()
      const ids = new Uint32Array(buf)
      entry.instances = ids.length === entry.mesh.numSplats ? ids : null
      entry.stale = r.headers.get('X-Gaussians-Stale') === '1'
    } catch {
      entry.instances = null
    }
  }

  /** Re-read the live instances of every model (after a segmentation edit / projection). */
  async refreshInstances(): Promise<void> {
    for (const e of this.entries.values()) { await this.fetchInstances(e); this.applyControl(e) }
  }

  private applyControl(entry: Entry) {
    const n = entry.mesh.numSplats
    const arr = entry.control.ensureCapacity(n)
    const ids = entry.instances
    const hl = this.highlighted
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      const id = ids ? ids[i] : 0
      arr[j] = hl != null && id === hl ? 255 : 0
      arr[j + 1] = 0
      arr[j + 2] = 0
      arr[j + 3] = this.hidden.has(id) ? 0 : 255
    }
    entry.control.count = n
    entry.control.needsUpdate = true
  }

  setModelVisible(name: string, visible: boolean) {
    const e = this.entries.get(name)
    if (!e) return
    e.visible = visible
    e.mesh.visible = visible
  }

  isModelVisible(name: string): boolean { return this.entries.get(name)?.visible ?? false }

  /** An instance switched off in the Instances list takes its splats with it (0 = the unsegmented remainder). */
  setInstanceVisible(instanceId: number, visible: boolean) {
    const had = this.hidden.has(instanceId)
    if (visible) this.hidden.delete(instanceId); else this.hidden.add(instanceId)
    if (had === !visible) return
    for (const e of this.entries.values()) this.applyControl(e)
  }

  setAllInstancesVisible(visible: boolean, instanceIds: Iterable<number>) {
    for (const id of instanceIds) { if (visible) this.hidden.delete(id); else this.hidden.add(id) }
    for (const e of this.entries.values()) this.applyControl(e)
  }

  highlight(instanceId: number | null) {
    if (this.highlighted === instanceId) return
    this.highlighted = instanceId
    for (const e of this.entries.values()) this.applyControl(e)
  }

  unload(name: string) {
    const e = this.entries.get(name)
    if (!e) return
    e.mesh.parent?.remove(e.mesh)
    e.mesh.dispose()
    e.control.dispose()
    this.entries.delete(name)
  }

  dispose() {
    for (const name of [...this.entries.keys()]) this.unload(name)
    if (this.spark) { this.scene.remove(this.spark); this.spark = null }
    this.hidden.clear()
    this.highlighted = null
  }
}
