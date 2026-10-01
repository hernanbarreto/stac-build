/**
 * PhysicsSandbox — gravity + bouncing spheres in the viewer (USER 2026-10-01: "agregar gravedad a la
 * escena, una esfera que pueda soltarse y si cae sobre algún objeto sólido debe rebotar, y si solo hay
 * nube de puntos no"; "y = 0 en principio es la base para la pelota").
 *
 * Rapier (WASM, @dimforge/rapier3d-compat). What is SOLID:
 *  - the floor plane y = 0 (the display frame is levelled to it);
 *  - every visible MESH under the roots it is given (placed library objects — the reference human —,
 *    ShapeR objects, Mesh results), as triangle-mesh colliders in world space.
 * The point cloud is NOT solid: a sphere falls through it. Colliders follow the scene: they are rebuilt
 * when a mesh appears, disappears, hides or moves (checked at most twice a second).
 */
import RAPIER from '@dimforge/rapier3d-compat'
import * as THREE from 'three'

const GRAVITY = -9.81                 // m/s², the display frame is metric with +Y up
const STEP = 1 / 60                   // fixed physics step (s)
const MAX_STEPS = 5                   // per frame — a stalled tab does not explode into a spiral
const RESYNC_S = 0.5                  // how often the scene's meshes are compared with the colliders
const FLOOR_HALF = 500                // the floor plane's half extent (m) — a slab whose top is y = 0
const LOST_Y = -50                    // a sphere below this fell out of the world and is removed

interface Ball { body: RAPIER.RigidBody; mesh: THREE.Mesh }
const THROW_WINDOW_S = 0.1             // the release velocity = the hand's motion over the last 0.1 s
const MAX_THROW_MS = 25                // a throw is capped at 25 m/s (a hard pitch)
interface Solid { collider: RAPIER.Collider; key: string }

function visibleInScene(o: THREE.Object3D): boolean {
    for (let p: THREE.Object3D | null = o; p; p = p.parent) if (!p.visible) return false
    return true
}

export class PhysicsSandbox {
    readonly group = new THREE.Group()
    private world: RAPIER.World | null = null
    private ready: Promise<void> | null = null
    private balls: Ball[] = []
    private solids = new Map<string, Solid>()   // mesh uuid → its collider and the pose it was built at
    private acc = 0
    private sinceSync = Infinity
    private roots: () => THREE.Object3D[] = () => []
    private held: { ball: Ball; trail: Array<{ t: number; p: THREE.Vector3 }> } | null = null

    constructor(scene: THREE.Scene) {
        this.group.name = 'physics-spheres'
        scene.add(this.group)
    }

    /** the scene objects whose meshes are solid (re-read at every sync) */
    setSolidRoots(fn: () => THREE.Object3D[]) { this.roots = fn }

    private init(): Promise<void> {
        if (!this.ready) {
            this.ready = RAPIER.init().then(() => {
                const w = new RAPIER.World({ x: 0, y: GRAVITY, z: 0 })
                const floor = w.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(0, -0.5, 0))
                w.createCollider(RAPIER.ColliderDesc.cuboid(FLOOR_HALF, 0.5, FLOOR_HALF).setRestitution(1.0), floor)
                this.world = w
            })
        }
        return this.ready
    }

    /** Make every visible mesh under the roots a static triangle-mesh collider; drop the ones gone. */
    private syncSolids() {
        const w = this.world
        if (!w) return
        const seen = new Set<string>()
        for (const root of this.roots()) {
            if (!root) continue
            root.updateMatrixWorld(true)
            root.traverse(o => {
                const m = o as THREE.Mesh
                if (!m.isMesh || !m.geometry || !visibleInScene(m)) return
                const pos = m.geometry.getAttribute('position') as THREE.BufferAttribute | undefined
                if (!pos || pos.count < 3) return
                const key = m.matrixWorld.elements.map(v => v.toFixed(5)).join(',') + `|${pos.count}`
                seen.add(m.uuid)
                const old = this.solids.get(m.uuid)
                if (old && old.key === key) return
                if (old) w.removeCollider(old.collider, false)
                const verts = new Float32Array(pos.count * 3)
                const v = new THREE.Vector3()
                for (let i = 0; i < pos.count; i++) {
                    v.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld)
                    verts[3 * i] = v.x; verts[3 * i + 1] = v.y; verts[3 * i + 2] = v.z
                }
                const idx = m.geometry.getIndex()
                const indices = idx ? new Uint32Array(idx.array as ArrayLike<number>)
                    : Uint32Array.from({ length: pos.count - (pos.count % 3) }, (_, i) => i)
                // restitution 1: the SPHERE decides how much it bounces (its Min combine rule wins)
                const collider = w.createCollider(RAPIER.ColliderDesc.trimesh(verts, indices).setRestitution(1.0))
                this.solids.set(m.uuid, { collider, key })
            })
        }
        for (const [uuid, s] of this.solids) {
            if (!seen.has(uuid)) { w.removeCollider(s.collider, false); this.solids.delete(uuid) }
        }
    }

    /** Drop a sphere of `diameter` metres with `restitution` (0 = dead, 1 = perfectly elastic) at `at`. */
    async drop(at: THREE.Vector3, diameter: number, restitution: number) {
        await this.init()
        const w = this.world!
        this.syncSolids(); this.sinceSync = 0
        const r = Math.max(0.01, diameter / 2)
        const body = w.createRigidBody(RAPIER.RigidBodyDesc.dynamic().setTranslation(at.x, at.y, at.z).setCcdEnabled(true))
        w.createCollider(RAPIER.ColliderDesc.ball(r).setRestitution(Math.min(1, Math.max(0, restitution)))
            .setRestitutionCombineRule(RAPIER.CoefficientCombineRule.Min).setFriction(0.5), body)
        const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 32, 16),
            new THREE.MeshStandardMaterial({ color: 0xe8552b, roughness: 0.45, metalness: 0.05 }))
        mesh.position.copy(at)
        mesh.name = 'physics-sphere'
        this.group.add(mesh)
        this.balls.push({ body, mesh })
    }

    /** Advance the simulation by the frame's real time (called from the render loop). */
    step(dt: number) {
        const w = this.world
        if (!w || !this.balls.length) return
        this.sinceSync += dt
        if (this.sinceSync >= RESYNC_S) { this.syncSolids(); this.sinceSync = 0 }
        this.acc = Math.min(this.acc + dt, STEP * MAX_STEPS)
        while (this.acc >= STEP) { w.step(); this.acc -= STEP }
        for (let i = this.balls.length - 1; i >= 0; i--) {
            const b = this.balls[i]
            const p = b.body.translation(), q = b.body.rotation()
            if (p.y < LOST_Y) { this.removeBall(i); continue }
            b.mesh.position.set(p.x, p.y, p.z)
            b.mesh.quaternion.set(q.x, q.y, q.z, q.w)
        }
    }

    private removeBall(i: number) {
        const b = this.balls[i]
        this.world?.removeRigidBody(b.body)
        this.group.remove(b.mesh)
        b.mesh.geometry.dispose(); (b.mesh.material as THREE.Material).dispose()
        this.balls.splice(i, 1)
    }

    /** The sphere under the ray (for grab-and-throw), or null. */
    pick(ray: THREE.Raycaster): THREE.Mesh | null {
        const hit = ray.intersectObjects(this.group.children, false)[0]
        return hit ? (hit.object as THREE.Mesh) : null
    }

    /** Hold a sphere: it follows the pointer (kinematic) until released. */
    grab(mesh: THREE.Mesh) {
        const ball = this.balls.find(b => b.mesh === mesh)
        if (!ball) return
        ball.body.setBodyType(RAPIER.RigidBodyType.KinematicPositionBased, true)
        this.held = { ball, trail: [{ t: performance.now() / 1000, p: mesh.position.clone() }] }
    }

    /** Move the held sphere to `p` (world). */
    moveHeld(p: THREE.Vector3) {
        if (!this.held) return
        const y = Math.max(p.y, (this.held.ball.mesh.geometry as THREE.SphereGeometry).parameters.radius)
        this.held.ball.body.setNextKinematicTranslation({ x: p.x, y, z: p.z })
        const t = performance.now() / 1000
        this.held.trail.push({ t, p: new THREE.Vector3(p.x, y, p.z) })
        while (this.held.trail.length > 2 && t - this.held.trail[0].t > THROW_WINDOW_S) this.held.trail.shift()
    }

    /** Let go: the sphere is dynamic again and leaves with the hand's velocity. */
    release() {
        if (!this.held) return
        const { ball, trail } = this.held
        this.held = null
        const a = trail[0], b = trail[trail.length - 1]
        const dt = b.t - a.t
        const v = dt > 1e-3 ? b.p.clone().sub(a.p).divideScalar(dt) : new THREE.Vector3()
        if (v.length() > MAX_THROW_MS) v.setLength(MAX_THROW_MS)
        ball.body.setBodyType(RAPIER.RigidBodyType.Dynamic, true)
        ball.body.setLinvel({ x: v.x, y: v.y, z: v.z }, true)
    }

    get holding() { return this.held !== null }

    get count() { return this.balls.length }

    clear() { this.held = null; for (let i = this.balls.length - 1; i >= 0; i--) this.removeBall(i) }

    dispose() {
        this.clear()
        this.group.removeFromParent()
        this.world?.free(); this.world = null; this.ready = null
        this.solids.clear()
    }
}
