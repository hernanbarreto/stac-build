// STAC-Builder — immersive assistant visualization layer.
//
// Draws and ANIMATES the geometry behind every spatial answer so the user SEES
// how a measurement was taken: clearance/distance lines with a growing reveal,
// object bounding boxes with dimension labels, plumb/level tilt arcs, position
// markers, and user-defined evaluation volumes (occupancy + fit placement).
//
// The numbers come from the deterministic Phase-5 tools (tool_measured); this
// layer only renders them. Self-contained: owns one THREE.Group and a small
// time-based animator list driven from the viewport render loop.
//
// Hernán Barreto - Ingerop IN3 Session IV - STAC

import * as THREE from 'three'

export interface TraceEntry {
    tool: string
    arguments: Record<string, unknown>
    result: Record<string, unknown>
}

export interface SceneObject {
    id: number
    label: string
    obb?: { transform: number[]; center: number[]; half_extents: number[] }
}

export interface UserVolume {
    volume_id: number
    name: string
    center: number[]
    size: number[]
    yaw_deg: number
}

const ACCENT = 0x4fd1ff      // cyan — measurement primary
const ACCENT2 = 0xe99d28     // amber — dimensions / emphasis
const GOOD = 0x3fb950        // green — free / fits
const BAD = 0xf85149         // red — occupied / no-fit
const VOL = 0x9d7bff         // violet — user volumes

interface Animator { update: (now: number) => boolean } // returns true when done

function labelSprite(text: string, color = '#e6edf3', bg = 'rgba(13,17,23,0.82)'): THREE.Sprite {
    const pad = 12, font = 44
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')!
    ctx.font = `600 ${font}px Inter, system-ui, sans-serif`
    const w = Math.ceil(ctx.measureText(text).width) + pad * 2
    const h = font + pad * 2
    canvas.width = w; canvas.height = h
    ctx.font = `600 ${font}px Inter, system-ui, sans-serif`
    ctx.fillStyle = bg
    roundRect(ctx, 0, 0, w, h, 14); ctx.fill()
    ctx.strokeStyle = 'rgba(79,209,255,0.55)'; ctx.lineWidth = 2
    roundRect(ctx, 1, 1, w - 2, h - 2, 13); ctx.stroke()
    ctx.fillStyle = color
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
    ctx.fillText(text, w / 2, h / 2 + 2)
    const tex = new THREE.CanvasTexture(canvas)
    tex.anisotropy = 4
    const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true })
    const sprite = new THREE.Sprite(mat)
    const scale = 0.0016
    sprite.scale.set(w * scale, h * scale, 1)
    sprite.renderOrder = 2000
    return sprite
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
    ctx.beginPath()
    ctx.moveTo(x + r, y)
    ctx.arcTo(x + w, y, x + w, y + h, r)
    ctx.arcTo(x + w, y + h, x, y + h, r)
    ctx.arcTo(x, y + h, x, y, r)
    ctx.arcTo(x, y, x + w, y, r)
    ctx.closePath()
}

function marker(pos: THREE.Vector3, color: number, radius = 0.02): THREE.Mesh {
    const m = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 20, 20),
        new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true }))
    m.position.copy(pos); m.renderOrder = 1999
    return m
}

function matrixFromStore(transform: number[]): THREE.Matrix4 {
    // store sends the world transform row-major (numpy ravel); THREE.fromArray
    // reads column-major, so fromArray gives Tᵀ and transpose() recovers T.
    return new THREE.Matrix4().fromArray(transform).transpose()
}

export class AssistantViz {
    readonly group: THREE.Group
    private animators: Animator[] = []
    private objects = new Map<number, SceneObject>()
    private volumes = new Map<number, THREE.Object3D>()

    constructor() {
        this.group = new THREE.Group()
        this.group.name = 'assistantGroup'
    }

    setObjects(objs: SceneObject[]) {
        this.objects.clear()
        for (const o of objs) this.objects.set(o.id, o)
    }

    update(now: number) {
        if (!this.animators.length) return
        this.animators = this.animators.filter((a) => !a.update(now))
    }

    /** Clear measurement overlays (keeps persistent user volumes). */
    clearMeasurements() {
        const keep = new Set(this.volumes.values())
        for (const c of [...this.group.children]) {
            if (!keep.has(c)) { this.group.remove(c); disposeDeep(c) }
        }
        this.animators = []
    }

    clearAll() {
        for (const c of [...this.group.children]) { this.group.remove(c); disposeDeep(c) }
        this.animators = []
        this.volumes.clear()
    }

    // ── measurement traces → animated geometry ──────────────────────
    /** Returns a bounding box of what was drawn (for optional camera framing). */
    visualizeTrace(trace: TraceEntry[]): THREE.Box3 | null {
        this.clearMeasurements()
        const box = new THREE.Box3()
        let drew = false
        const now = performance.now()
        for (const e of trace || []) {
            const b = this.drawEntry(e, now)
            if (b) { box.union(b); drew = true }
        }
        return drew ? box : null
    }

    private objCenter(id: number | undefined): THREE.Vector3 | null {
        if (id == null) return null
        const o = this.objects.get(id)
        if (!o?.obb) return null
        return new THREE.Vector3().fromArray(o.obb.center)
    }

    private drawEntry(e: TraceEntry, now: number): THREE.Box3 | null {
        const r = e.result || {}
        const a = e.arguments || {}
        if (r.insufficient_data) return null
        switch (e.tool) {
            case 'get_clearance':
            case 'get_distance':
                return this.drawSpan(r, a, now)
            case 'get_object_size':
            case 'get_object_volume':
            case 'get_span':
                return this.drawObjectBox(a.id as number, r, now)
            case 'get_position':
                return this.drawPosition(a.id as number, r, now)
            case 'get_plumb':
            case 'get_level':
                return this.drawTilt(a.id as number, r, e.tool, now)
            case 'evaluate_volume':
            case 'objects_in_volume':
                return this.drawVolumeEval(a, r, now)
            case 'fits_in_volume':
                return this.drawFit(a, r, now)
            default:
                return null
        }
    }

    private drawSpan(r: Record<string, unknown>, a: Record<string, unknown>, now: number): THREE.Box3 | null {
        let pA = Array.isArray(r.point_a_m) ? new THREE.Vector3().fromArray(r.point_a_m as number[]) : null
        let pB = Array.isArray(r.point_b_m) ? new THREE.Vector3().fromArray(r.point_b_m as number[]) : null
        if (!pA || !pB) { pA = this.objCenter(a.id1 as number); pB = this.objCenter(a.id2 as number) }
        if (!pA || !pB) return null
        const dist = (r.clearance_m ?? r.distance_m) as number
        const g = new THREE.Group()
        g.add(marker(pA, ACCENT)); g.add(marker(pB, ACCENT))
        const lineGeom = new THREE.BufferGeometry().setFromPoints([pA.clone(), pA.clone()])
        const line = new THREE.Line(lineGeom, new THREE.LineBasicMaterial({ color: ACCENT, depthTest: false, transparent: true }))
        line.renderOrder = 1998
        g.add(line)
        const label = labelSprite(`${(dist ?? pA.distanceTo(pB)).toFixed(3)} m`, '#4fd1ff')
        label.position.copy(pA.clone().lerp(pB, 0.5))
        label.visible = false
        g.add(label)
        this.group.add(g)
        // animate: line grows A→B over 500ms, then label fades in
        this.animators.push(revealLine(line, pA, pB, label, now, 550))
        return new THREE.Box3().setFromPoints([pA, pB])
    }

    private drawObjectBox(id: number | undefined, r: Record<string, unknown>, now: number): THREE.Box3 | null {
        if (id == null) return null
        const o = this.objects.get(id)
        if (!o?.obb) return null
        const M = matrixFromStore(o.obb.transform)
        const half = o.obb.half_extents
        const box = new THREE.BoxGeometry(half[0] * 2, half[1] * 2, half[2] * 2)
        const edges = new THREE.LineSegments(
            new THREE.EdgesGeometry(box),
            new THREE.LineBasicMaterial({ color: ACCENT2, depthTest: false, transparent: true }))
        edges.applyMatrix4(M)
        edges.renderOrder = 1998
        const g = new THREE.Group(); g.add(edges)
        // dimension label
        const dims = (r.width_m != null)
            ? `${r.width_m}×${r.height_m}×${r.depth_m} m`
            : (r.bbox_volume_m3 != null ? `${r.bbox_volume_m3} m³` : `${r.span_m} m`)
        const center = new THREE.Vector3().fromArray(o.obb.center)
        const label = labelSprite(`${o.label}: ${dims}`, '#f0c674')
        label.position.copy(center).y += half[1] + 0.15
        g.add(label)
        this.group.add(g)
        this.animators.push(growScale(edges, now, 420))
        const bb = new THREE.Box3().setFromObject(edges)
        return bb
    }

    private drawPosition(id: number | undefined, r: Record<string, unknown>, now: number): THREE.Box3 | null {
        const pos = Array.isArray(r.position_m)
            ? new THREE.Vector3().fromArray(r.position_m as number[])
            : this.objCenter(id)
        if (!pos) return null
        const g = new THREE.Group()
        const m = marker(pos, ACCENT, 0.03); g.add(m)
        const label = labelSprite(`(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)}) m`)
        label.position.copy(pos).y += 0.12
        g.add(label)
        this.group.add(g)
        this.animators.push(pulse(m, now, 900))
        return new THREE.Box3().setFromCenterAndSize(pos, new THREE.Vector3(0.3, 0.3, 0.3))
    }

    private drawTilt(id: number | undefined, r: Record<string, unknown>, tool: string, now: number): THREE.Box3 | null {
        const c = this.objCenter(id)
        if (!c) return null
        const o = this.objects.get(id!)
        const h = o?.obb ? o.obb.half_extents[1] : 1
        const len = Math.max(0.6, h)
        const up = new THREE.Vector3(0, 1, 0)
        const deg = (r.plumb_deviation_deg ?? r.level_deviation_deg) as number
        // reference (true vertical/horizontal) + measured axis tilted by deg
        const g = new THREE.Group()
        const ref = tool === 'get_plumb' ? up.clone() : new THREE.Vector3(1, 0, 0)
        const refLine = dashed(c, c.clone().addScaledVector(ref, len), 0x8899aa)
        const rad = THREE.MathUtils.degToRad(deg || 0)
        const tilt = ref.clone().applyAxisAngle(new THREE.Vector3(0, 0, 1), rad).normalize()
        const measLine = solid(c, c.clone().addScaledVector(tilt, len), deg > 3 ? BAD : GOOD)
        g.add(refLine); g.add(measLine)
        const label = labelSprite(`${(deg ?? 0).toFixed(2)}° · ${(r.deviation_mm_per_m ?? 0)} mm/m`,
            deg > 3 ? '#f85149' : '#3fb950')
        label.position.copy(c).addScaledVector(tilt, len).y += 0.1
        g.add(label)
        this.group.add(g)
        this.animators.push(pulse(measLine, now, 900))
        return new THREE.Box3().setFromObject(g)
    }

    private volumeFrame(a: Record<string, unknown>): { M: THREE.Matrix4; size: number[] } | null {
        let center = a.center as number[] | undefined
        let size = a.size as number[] | undefined
        let yaw = (a.yaw_deg as number) || 0
        if ((!center || !size) && a.volume_id != null) {
            const v = this.volumesMeta.get(a.volume_id as number)
            if (v) { center = v.center; size = v.size; yaw = v.yaw_deg }
        }
        if (!center || !size) return null
        const M = new THREE.Matrix4().makeRotationY(THREE.MathUtils.degToRad(yaw))
        M.setPosition(new THREE.Vector3().fromArray(center))
        return { M, size }
    }

    private volumesMeta = new Map<number, UserVolume>()

    private drawVolumeEval(a: Record<string, unknown>, r: Record<string, unknown>, now: number): THREE.Box3 | null {
        const f = this.volumeFrame(a)
        if (!f) return null
        const g = new THREE.Group()
        const freeFrac = (r.free_fraction as number) ?? 1
        const boxMesh = translucentBox(f.size, freeFrac > 0.5 ? GOOD : BAD, 0.12)
        boxMesh.applyMatrix4(f.M)
        const edges = boxEdges(f.size, VOL); edges.applyMatrix4(f.M)
        g.add(boxMesh); g.add(edges)
        // highlight objects inside
        const inside = (r.objects_inside as Array<{ id: number }> | undefined)
            ?? (r.objects as Array<{ id: number }> | undefined) ?? []
        for (const oi of inside) {
            const o = this.objects.get(oi.id)
            if (o?.obb) {
                const e = boxEdges(o.obb.half_extents.map((x) => x * 2), ACCENT2)
                e.applyMatrix4(matrixFromStore(o.obb.transform))
                g.add(e)
            }
        }
        const center = new THREE.Vector3().setFromMatrixPosition(f.M)
        const txt = r.free_volume_m3 != null
            ? `free ${r.free_volume_m3} m³ · ${Math.round((freeFrac) * 100)}% · ${inside.length} obj`
            : `${inside.length} objects inside`
        const label = labelSprite(txt, freeFrac > 0.5 ? '#3fb950' : '#f85149')
        label.position.copy(center).y += f.size[1] / 2 + 0.15
        g.add(label)
        this.group.add(g)
        this.animators.push(growScale(boxMesh, now, 500))
        return new THREE.Box3().setFromObject(edges)
    }

    private drawFit(a: Record<string, unknown>, r: Record<string, unknown>, now: number): THREE.Box3 | null {
        const f = this.volumeFrame(a)
        if (!f) return null
        const g = new THREE.Group()
        const edges = boxEdges(f.size, VOL); edges.applyMatrix4(f.M); g.add(edges)
        const fits = !!r.fits
        if (fits && Array.isArray(r.placement_box_local_m) && Array.isArray(r.item_size_m)) {
            const item = r.item_size_m as number[]
            const p = r.placement_box_local_m as number[]
            // placement is box-local origin (0..size); convert to centre in local coords
            const localCenter = new THREE.Vector3(
                p[0] + item[0] / 2 - f.size[0] / 2,
                p[1] + item[1] / 2 - f.size[1] / 2,
                p[2] + item[2] / 2 - f.size[2] / 2)
            const itemMesh = translucentBox(item, GOOD, 0.35)
            itemMesh.position.copy(localCenter)
            itemMesh.applyMatrix4(f.M)
            g.add(itemMesh)
            const label = labelSprite(`fits: ${item.map((x) => x.toFixed(2)).join('×')} m`, '#3fb950')
            label.position.copy(new THREE.Vector3().setFromMatrixPosition(f.M)).y += f.size[1] / 2 + 0.15
            g.add(label)
            this.animators.push(growScale(itemMesh, now, 500))
        } else {
            const label = labelSprite(`does not fit`, '#f85149')
            label.position.copy(new THREE.Vector3().setFromMatrixPosition(f.M)).y += f.size[1] / 2 + 0.15
            g.add(label)
        }
        this.group.add(g)
        return new THREE.Box3().setFromObject(edges)
    }

    // ── persistent user volumes (placed from the panel) ─────────────
    addVolume(v: UserVolume) {
        this.removeVolume(v.volume_id)
        this.volumesMeta.set(v.volume_id, v)
        const M = new THREE.Matrix4().makeRotationY(THREE.MathUtils.degToRad(v.yaw_deg))
        M.setPosition(new THREE.Vector3().fromArray(v.center))
        const g = new THREE.Group()
        g.name = `userVolume_${v.volume_id}`
        const box = translucentBox(v.size, VOL, 0.08); box.applyMatrix4(M)
        const edges = boxEdges(v.size, VOL); edges.applyMatrix4(M)
        const label = labelSprite(v.name, '#c4b5ff')
        label.position.copy(new THREE.Vector3().fromArray(v.center)).y += v.size[1] / 2 + 0.12
        g.add(box); g.add(edges); g.add(label)
        this.group.add(g)
        this.volumes.set(v.volume_id, g)
    }

    removeVolume(id: number) {
        const g = this.volumes.get(id)
        if (g) { this.group.remove(g); disposeDeep(g); this.volumes.delete(id) }
        this.volumesMeta.delete(id)
    }
}

// ── geometry helpers ────────────────────────────────────────────────
function boxEdges(size: number[], color: number): THREE.LineSegments {
    const e = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(size[0], size[1], size[2])),
        new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true }))
    e.renderOrder = 1998
    return e
}

function translucentBox(size: number[], color: number, opacity: number): THREE.Mesh {
    const m = new THREE.Mesh(
        new THREE.BoxGeometry(size[0], size[1], size[2]),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false }))
    m.renderOrder = 1997
    return m
}

function solid(a: THREE.Vector3, b: THREE.Vector3, color: number): THREE.Line {
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]),
        new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true }))
    l.renderOrder = 1998
    return l
}

function dashed(a: THREE.Vector3, b: THREE.Vector3, color: number): THREE.Line {
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]),
        new THREE.LineDashedMaterial({ color, dashSize: 0.08, gapSize: 0.05, depthTest: false, transparent: true }))
    l.computeLineDistances(); l.renderOrder = 1998
    return l
}

function disposeDeep(o: THREE.Object3D) {
    o.traverse((c) => {
        const any = c as unknown as { geometry?: THREE.BufferGeometry; material?: THREE.Material | THREE.Material[] }
        any.geometry?.dispose()
        const m = any.material
        if (Array.isArray(m)) m.forEach((x) => x.dispose())
        else m?.dispose()
    })
}

// ── animators (return true when finished) ───────────────────────────
function revealLine(line: THREE.Line, a: THREE.Vector3, b: THREE.Vector3,
    label: THREE.Sprite, start: number, dur: number): Animator {
    return {
        update: (now) => {
            const t = Math.min(1, (now - start) / dur)
            const cur = a.clone().lerp(b, t)
            line.geometry.setFromPoints([a, cur])
            if (t >= 1) { label.visible = true; return true }
            return false
        },
    }
}

function growScale(obj: THREE.Object3D, start: number, dur: number): Animator {
    const base = obj.position.clone()
    return {
        update: (now) => {
            const t = Math.min(1, (now - start) / dur)
            const s = 0.001 + easeOut(t)
            obj.scale.setScalar(s)
            // keep centred: BoxGeometry is centred at origin, scaling about it is fine
            obj.position.copy(base)
            return t >= 1
        },
    }
}

function pulse(obj: THREE.Object3D, start: number, dur: number): Animator {
    const mat = (obj as THREE.Mesh).material as THREE.Material & { opacity?: number }
    return {
        update: (now) => {
            const t = (now - start) / dur
            if (mat && 'opacity' in mat) mat.opacity = 0.55 + 0.45 * Math.abs(Math.sin(t * Math.PI * 2))
            return t >= 3 // ~3 pulses then settle
        },
    }
}

function easeOut(t: number): number { return 1 - Math.pow(1 - t, 3) }
