/**
 * SceneLights — the scene is lit by the light sources the SEGMENTATION found (USER 2026-10-16: "las fuentes
 * de luz deben ser tantas y estar ubicadas como luz natural o artificial haya en los segmentos"):
 *  - every instance whose label names a lamp / luminaire becomes a POINT light at its box — a ceiling fixture's
 *    light hangs just under its bottom face, so its own shell (points or a generated mesh) does not block it;
 *  - every window / skylight becomes a SPOT of daylight entering through it, aimed into the room (toward the
 *    centre of everything segmented), the largest ones casting shadows;
 *  - the Display "light" slider scales all of them, together with the ambient and the shadow sun.
 * Units are three's physical ones (r155+): a point light in candela with inverse-square decay. Nothing here
 * measures — the labels say what is a source, the boxes where; the amounts are a look, declared below. The
 * point cloud is not lit by these (its shader has no normals); the meshes are.
 */
import * as THREE from 'three'

const ARTIFICIAL = /\b(lamp|lamps|light|lights|lighting|luminaire|luminaires|fluorescent|fluorescents|bulb|bulbs|led|leds|spotlight|spotlights|downlight|downlights|chandelier|sconce|lantern|floodlight|floodlights)\b/i
const NATURAL = /\b(window|windows|skylight|skylights|glazing|rooflight|clerestory)\b/i
const NOT_SOURCES = /\b(switch|switches|socket|sockets|sign|signs|signal|signals|pole|poles|traffic|headlight|taillight|reflector|frame|sill|blind|blinds|curtain|curtains)\b/i
const LAMP_CANDELA = 250                 // one room lamp (≈ 3000 lm omnidirectional)
const DAYLIGHT_CANDELA_PER_M2 = 1500     // daylight through one square metre of glazing
const DAYLIGHT_CANDELA_MAX = 9000
const LAMP_DROP_M = 0.15                 // the light hangs this far under a ceiling fixture's centre
const CEILING_MIN_Y = 1.8                // a fixture above this height is a ceiling one
const SPOT_ANGLE = Math.PI / 3
const SPOT_PENUMBRA = 0.6
const MAX_SHADOW_SPOTS = 4
const MAX_LIGHTS = 48

export interface LightRecord {
    instanceId: number
    label: string
    kind: 'lamp' | 'daylight'
    position: [number, number, number]
    candela: number
    shadows: boolean
}

export class SceneLights {
    readonly group = new THREE.Group()
    readonly records: LightRecord[] = []
    private scale = 1

    constructor(scene: THREE.Scene) {
        this.group.name = 'segment-lights'
        scene.add(this.group)
    }

    /** the Display light slider: every source × scale */
    setScale(v: number) {
        this.scale = v
        for (const o of this.group.children) {
            const l = o as THREE.Light
            if ((l as THREE.Light).isLight && typeof l.userData.base === 'number') l.intensity = l.userData.base * v
        }
    }

    clear() {
        for (const o of [...this.group.children]) {
            const l = o as THREE.SpotLight
            if (l.shadow && l.shadow.map) l.shadow.map.dispose()
            this.group.remove(o)
        }
        this.records.length = 0
    }

    /** Rebuild from the segmentation's instances — the same OBB list the viewer draws. */
    rebuild(instances: Array<Record<string, unknown>>) {
        this.clear()
        type Box = { inst: Record<string, unknown>; centre: THREE.Vector3; half: number[]; rot?: number[][]; label: string; iid: number }
        const boxes: Box[] = []
        for (const inst of instances) {
            const obb = inst.obb as Record<string, unknown> | undefined
            const c = obb?.center as number[] | undefined
            if (!obb || !c || c.length !== 3) continue
            const half = ((obb.half_extents ?? obb.halfExtents) as number[] | undefined) ?? [0.2, 0.2, 0.2]
            boxes.push({ inst, centre: new THREE.Vector3(c[0], c[1], c[2]), half: half.map(Number), rot: obb.rotation as number[][] | undefined,
                         label: String(inst.label || ''), iid: Number(inst.instance_id || inst.id || 0) })
        }
        if (!boxes.length) return
        const roomCentre = new THREE.Vector3()
        for (const b of boxes) roomCentre.add(b.centre)
        roomCentre.divideScalar(boxes.length)

        const lamps: Box[] = []
        const windows: Array<{ b: Box; area: number; normal: THREE.Vector3 }> = []
        for (const b of boxes) {
            if (NOT_SOURCES.test(b.label)) continue
            if (NATURAL.test(b.label)) {
                // the pane's normal is the box axis of its smallest extent (the columns of `rotation`
                // are the box axes in the world, as renderOBBs applies it), turned into the room
                let k = 0
                for (let i = 1; i < 3; i++) if (b.half[i] < b.half[k]) k = i
                const r = b.rot
                const n = r && r.length === 3 ? new THREE.Vector3(r[0][k], r[1][k], r[2][k]) : new THREE.Vector3(1, 0, 0)
                if (n.lengthSq() < 1e-6) n.set(1, 0, 0)
                n.normalize()
                if (n.dot(new THREE.Vector3().subVectors(roomCentre, b.centre)) < 0) n.negate()
                const others = [0, 1, 2].filter(i => i !== k)
                windows.push({ b, area: 4 * b.half[others[0]] * b.half[others[1]], normal: n })
            } else if (ARTIFICIAL.test(b.label)) {
                lamps.push(b)
            }
        }
        windows.sort((a, c) => c.area - a.area)

        let shadowSpots = 0
        for (const w of windows) {
            if (this.group.children.length >= MAX_LIGHTS) break
            const candela = Math.min(DAYLIGHT_CANDELA_MAX, DAYLIGHT_CANDELA_PER_M2 * Math.max(0.1, w.area))
            const spot = new THREE.SpotLight(0xf4f8ff, candela * this.scale, 0, SPOT_ANGLE, SPOT_PENUMBRA, 2)
            spot.name = `daylight-${w.b.iid}`
            spot.userData.base = candela
            spot.position.copy(w.b.centre).addScaledVector(w.normal, 0.1)      // just inside the pane: the glass never shadows it
            spot.target.position.copy(w.b.centre).addScaledVector(w.normal, 4.0)
            const withShadow = shadowSpots < MAX_SHADOW_SPOTS
            if (withShadow) {
                spot.castShadow = true
                spot.shadow.mapSize.set(1024, 1024)
                spot.shadow.bias = -0.0005
                spot.shadow.normalBias = 0.02
                spot.shadow.camera.near = 0.1
                spot.shadow.camera.far = 25
                shadowSpots += 1
            }
            this.group.add(spot, spot.target)
            this.records.push({ instanceId: w.b.iid, label: w.b.label, kind: 'daylight', position: spot.position.toArray() as [number, number, number],
                                candela, shadows: withShadow })
        }
        for (const b of lamps) {
            if (this.group.children.length >= MAX_LIGHTS) break
            const light = new THREE.PointLight(0xfff1dc, LAMP_CANDELA * this.scale, 0, 2)
            light.name = `lamp-${b.iid}`
            light.userData.base = LAMP_CANDELA
            light.position.copy(b.centre)
            if (b.centre.y > CEILING_MIN_Y) light.position.y -= LAMP_DROP_M
            light.castShadow = false        // a point light's shadow is six passes over the scene mesh; the sun and the windows cast
            this.group.add(light)
            this.records.push({ instanceId: b.iid, label: b.label, kind: 'lamp', position: light.position.toArray() as [number, number, number],
                                candela: LAMP_CANDELA, shadows: false })
        }
    }
}
