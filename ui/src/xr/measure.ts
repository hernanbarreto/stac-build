// STAC-Builder — shared measurement kit for the mobile XR engines.
//
// One implementation of the metric measurement tools (distance / angle /
// volume box) used by BOTH engines (8th Wall fallback and WebXR/ARKit via
// Variant Launch): points arrive in METRIC MODEL SPACE, so values stay true
// at any display scale.
//
// Hernán Barreto - Ingerop IN3 Session IV - STAC

import * as THREE from 'three'
import type { Tool } from './engine-types'

export class MeasureKit {
  private pending: THREE.Vector3[] = []

  constructor(private group: THREE.Group) {}

  clear() {
    this.pending = []
    while (this.group.children.length) this.group.children.pop()
  }

  resetPending() { this.pending = [] }

  private fmt(m: number) {
    return m >= 1 ? `${m.toFixed(2)} m` : `${(m * 100).toFixed(1)} cm`
  }

  private marker(p: THREE.Vector3, color = 0xf0a839) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 12),
                             new THREE.MeshBasicMaterial({ color }))
    s.position.copy(p)
    this.group.add(s)
  }

  private line(points: THREE.Vector3[], color = 0xf0a839) {
    this.group.add(new THREE.Line(
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
    this.group.add(sp)
  }

  /** Add a picked METRIC model-space point for the given tool. */
  addPoint(tool: Tool, p: THREE.Vector3) {
    if (tool === 'dist') {
      this.pending.push(p); this.marker(p)
      if (this.pending.length === 2) {
        const [a, b] = this.pending
        this.line([a, b])
        this.label(this.fmt(a.distanceTo(b)),
                   a.clone().add(b).multiplyScalar(0.5))
        this.pending = []
      }
    } else if (tool === 'angle') {
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
    } else if (tool === 'vol') {
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
        this.group.add(box)
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo),
          new THREE.LineBasicMaterial({ color: 0x3fb950 }))
        edges.position.copy(box.position)
        this.group.add(edges)
        this.label(
          `${size.x.toFixed(2)}×${size.z.toFixed(2)}×${size.y.toFixed(2)} m = `
          + `${vol.toFixed(2)} m³`,
          box.position.clone().setY(min.y + size.y + 0.1))
        this.pending = []
      }
    }
  }
}
