/**
 * PhysicsSandbox — gravity + bouncing spheres in the viewer (USER 2026-10-01: "agregar gravedad a la
 * escena, una esfera que pueda soltarse y si cae sobre algún objeto sólido debe rebotar"; then "los puntos
 * también son superficie" and "tengo el piso con desnivel … si no hay piso debe caer y desaparecer, si hay
 * piso debe ir copiando el nivel del piso, siempre el nivel mínimo de la nube").
 *
 * Rapier (WASM, @dimforge/rapier3d-compat). What is SOLID:
 *  - NO floor plane: the floor is the cloud's own (it has steps and slopes); where there is no surface the
 *    sphere falls, and below the cloud's lowest point it is gone;
 *  - every visible MESH under the roots it is given (placed library objects — the reference human —,
 *    ShapeR objects, Mesh results), as triangle-mesh colliders in world space.
 *  - the POINT CLOUD too (USER 2026-10-01: "los puntos también son superficie"): around each sphere the
 *    loaded points are binned into VOXEL cells and every occupied cell near it becomes a thin solid PLATE
 *    lying on the local surface (normal fitted to the points of the cell and its 26 neighbours), so a floor
 *    of points is a smooth floor and a wall a smooth wall (cubes deflected the sphere at their edges) —
 *    only near the spheres, so millions of points cost nothing until a sphere gets there.
 * Mesh colliders follow the scene: rebuilt when a mesh appears, disappears, hides or moves (checked at
 * most twice a second). Hidden points (the cloud switched off, a culled node) are not solid.
 */
import RAPIER from '@dimforge/rapier3d-compat'
import * as THREE from 'three'

const GRAVITY = -9.81                 // m/s², the display frame is metric with +Y up
const STEP = 1 / 60                   // fixed physics step (s)
const MAX_STEPS = 5                   // per frame — a stalled tab does not explode into a spiral
const RESYNC_S = 0.5                  // how often the scene's meshes are compared with the colliders
const RESYNC_CELLS_S = 0.05           // how often the point cells around the spheres are refreshed
const LOST_BELOW = 1.0                // m under the cloud's lowest point: the sphere fell out and is removed
const VOXEL = 0.03                    // m — the point cloud as solid cells of this size
const NEAR = 0.25                     // m — cells this far beyond a sphere's surface are made solid
const PLATE_HALF = VOXEL * 0.75       // a cell's plate overlaps its neighbours' — no gap to fall through
const PLATE_THICK = VOXEL / 3         // and is thin: the surface, not a cube

interface Ball { body: RAPIER.RigidBody; mesh: THREE.Mesh }
const THROW_WINDOW_S = 0.1             // the release velocity = the hand's motion over the last 0.1 s
const MAX_THROW_MS = 25                // a throw is capped at 25 m/s (a hard pitch)
interface Solid { collider: RAPIER.Collider; key: string }

/** Unit eigenvector of the smallest eigenvalue of a symmetric 3×3 matrix (Jacobi rotations). */
function smallestEigenvector(A: number[][]): THREE.Vector3 {
    const a = A.map(r => r.slice())
    const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for (let sweep = 0; sweep < 32; sweep++) {
        const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2])
        if (off < 1e-15) break
        for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
            if (Math.abs(a[p][q]) < 1e-18) continue
            const th = (a[q][q] - a[p][p]) / (2 * a[p][q])
            const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1))
            const c = 1 / Math.sqrt(t * t + 1), s = t * c
            for (let k = 0; k < 3; k++) {
                const akp = a[k][p], akq = a[k][q]
                a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq
            }
            for (let k = 0; k < 3; k++) {
                const apk = a[p][k], aqk = a[q][k]
                a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk
            }
            for (let k = 0; k < 3; k++) {
                const vkp = V[k][p], vkq = V[k][q]
                V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq
            }
        }
    }
    const m = [0, 1, 2].reduce((b, i) => (a[i][i] < a[b][b] ? i : b), 0)
    return new THREE.Vector3(V[0][m], V[1][m], V[2][m]).normalize()
}

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
    private sinceCells = Infinity
    private roots: () => THREE.Object3D[] = () => []
    private held: { ball: Ball; trail: Array<{ t: number; p: THREE.Vector3 }> } | null = null
    private pointRoots: () => THREE.Object3D[] = () => []
    private occupied = new Map<string, Float64Array>() // voxel → [n, Σx, Σy, Σz, Σxx, Σyy, Σzz, Σxy, Σxz, Σyz]
    private binned = new WeakSet<THREE.BufferGeometry>() // point geometries already binned
    private cells = new Map<string, RAPIER.Collider>() // the solid cells around the spheres

    constructor(scene: THREE.Scene) {
        this.group.name = 'physics-spheres'
        scene.add(this.group)
    }

    /** the scene objects whose meshes are solid (re-read at every sync) */
    setSolidRoots(fn: () => THREE.Object3D[]) { this.roots = fn }

    /** the point clouds (Potree octree group, a plain THREE.Points) whose points are solid */
    setPointRoots(fn: () => THREE.Object3D[]) { this.pointRoots = fn }

    /** Bin the visible point nodes that reach the region [lo, hi] (world) into the voxel set. */
    private binPointsNear(lo: THREE.Vector3, hi: THREE.Vector3) {
        const region = new THREE.Box3(lo, hi)
        const box = new THREE.Box3(), v = new THREE.Vector3()
        for (const root of this.pointRoots()) {
            if (!root) continue
            root.updateMatrixWorld(true)
            root.traverse(o => {
                const pts = o as THREE.Points
                if (!pts.isPoints || !pts.geometry || this.binned.has(pts.geometry) || !visibleInScene(pts)) return
                const pos = pts.geometry.getAttribute('position') as THREE.BufferAttribute | undefined
                if (!pos || !pos.count) return
                if (!pts.geometry.boundingBox) pts.geometry.computeBoundingBox()
                box.copy(pts.geometry.boundingBox!).applyMatrix4(pts.matrixWorld)
                if (!box.intersectsBox(region)) return
                for (let i = 0; i < pos.count; i++) {
                    v.fromBufferAttribute(pos, i).applyMatrix4(pts.matrixWorld)
                    const key = `${Math.floor(v.x / VOXEL)},${Math.floor(v.y / VOXEL)},${Math.floor(v.z / VOXEL)}`
                    let a = this.occupied.get(key)
                    if (!a) { a = new Float64Array(10); this.occupied.set(key, a) }
                    a[0] += 1; a[1] += v.x; a[2] += v.y; a[3] += v.z
                    a[4] += v.x * v.x; a[5] += v.y * v.y; a[6] += v.z * v.z
                    a[7] += v.x * v.y; a[8] += v.x * v.z; a[9] += v.y * v.z
                }
                this.binned.add(pts.geometry)
            })
        }
    }

    /** Make the occupied cells near every sphere solid; release the ones no sphere is near any more. */
    private syncCells() {
        const w = this.world
        if (!w) return
        const want = new Set<string>()
        for (const b of this.balls) {
            const p = b.body.translation()
            const lin = b.body.linvel()
            const r = (b.mesh.geometry as THREE.SphereGeometry).parameters.radius
            const reach = r + NEAR + Math.hypot(lin.x, lin.y, lin.z) * RESYNC_CELLS_S
            const lo = new THREE.Vector3(p.x - reach, p.y - reach, p.z - reach)
            const hi = new THREE.Vector3(p.x + reach, p.y + reach, p.z + reach)
            this.binPointsNear(lo, hi)
            const i0 = Math.floor(lo.x / VOXEL), i1 = Math.floor(hi.x / VOXEL)
            const j0 = Math.floor(lo.y / VOXEL), j1 = Math.floor(hi.y / VOXEL)
            const k0 = Math.floor(lo.z / VOXEL), k1 = Math.floor(hi.z / VOXEL)
            for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) for (let k = k0; k <= k1; k++) {
                const key = `${i},${j},${k}`
                if (this.occupied.has(key)) want.add(key)
            }
        }
        for (const [key, c] of this.cells) if (!want.has(key)) { w.removeCollider(c, false); this.cells.delete(key) }
        for (const key of want) {
            if (this.cells.has(key)) continue
            const plate = this.cellPlate(key)
            const desc = RAPIER.ColliderDesc.cuboid(PLATE_HALF, PLATE_HALF, PLATE_THICK / 2)
                .setTranslation(plate.c.x, plate.c.y, plate.c.z)
                .setRotation({ x: plate.q.x, y: plate.q.y, z: plate.q.z, w: plate.q.w }).setRestitution(1.0)
            this.cells.set(key, w.createCollider(desc))
        }
    }

    /** A cell's plate: centred on its points' centroid, its thin axis along the normal of the points of
     *  the cell and its 26 neighbours (the smallest-variance direction of their covariance). */
    private cellPlate(key: string): { c: THREE.Vector3; q: THREE.Quaternion } {
        const [i, j, k] = key.split(',').map(Number)
        const own = this.occupied.get(key)!
        const c = new THREE.Vector3(own[1] / own[0], own[2] / own[0], own[3] / own[0])
        const S = new Float64Array(10)
        for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) for (let dk = -1; dk <= 1; dk++) {
            const a = this.occupied.get(`${i + di},${j + dj},${k + dk}`)
            if (a) for (let t = 0; t < 10; t++) S[t] += a[t]
        }
        const n = S[0], mx = S[1] / n, my = S[2] / n, mz = S[3] / n
        const C = [[S[4] / n - mx * mx, S[7] / n - mx * my, S[8] / n - mx * mz],
                   [S[7] / n - mx * my, S[5] / n - my * my, S[9] / n - my * mz],
                   [S[8] / n - mx * mz, S[9] / n - my * mz, S[6] / n - mz * mz]]
        const normal = smallestEigenvector(C)
        if (n < 3 || !isFinite(normal.x)) normal.set(0, 1, 0)            // too few points: a level plate
        const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal)
        return { c, q }
    }

    private init(): Promise<void> {
        if (!this.ready) {
            this.ready = RAPIER.init().then(() => {
                this.world = new RAPIER.World({ x: 0, y: GRAVITY, z: 0 })
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
        this.syncCells(); this.sinceCells = 0
    }

    /** Advance the simulation by the frame's real time (called from the render loop). */
    step(dt: number) {
        const w = this.world
        if (!w || !this.balls.length) return
        this.sinceSync += dt
        if (this.sinceSync >= RESYNC_S) { this.syncSolids(); this.sinceSync = 0 }
        this.sinceCells += dt
        if (this.sinceCells >= RESYNC_CELLS_S) { this.syncCells(); this.sinceCells = 0 }
        this.acc = Math.min(this.acc + dt, STEP * MAX_STEPS)
        while (this.acc >= STEP) { w.step(); this.acc -= STEP }
        for (let i = this.balls.length - 1; i >= 0; i--) {
            const b = this.balls[i]
            const p = b.body.translation(), q = b.body.rotation()
            if (b !== this.held?.ball && p.y < this.lostY()) { this.removeBall(i); continue }
            b.mesh.position.set(p.x, p.y, p.z)
            b.mesh.quaternion.set(q.x, q.y, q.z, q.w)
        }
    }

    private removeBall(i: number) {
        const b = this.balls[i]
        if (this.held?.ball === b) this.held = null      // never leave a hand holding a sphere that is gone
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
        this.held.ball.body.setNextKinematicTranslation({ x: p.x, y: p.y, z: p.z })
        const t = performance.now() / 1000
        this.held.trail.push({ t, p: p.clone() })
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

    /** Below this a sphere has fallen out of the scene: LOST_BELOW under the lowest visible point (or
     *  mesh) of the scene; with nothing in the scene, under the place it was dropped from. */
    private lostY(): number {
        const box = new THREE.Box3()
        for (const root of [...this.pointRoots(), ...this.roots()]) if (root && visibleInScene(root)) box.expandByObject(root)
        const lowest = box.isEmpty() ? Math.min(...this.balls.map(b => b.mesh.position.y), 0) - 50 : box.min.y
        return lowest - LOST_BELOW
    }

    get count() { return this.balls.length }

    clear() {
        this.held = null
        for (let i = this.balls.length - 1; i >= 0; i--) this.removeBall(i)
        for (const c of this.cells.values()) this.world?.removeCollider(c, false)
        this.cells.clear()
    }

    dispose() {
        this.clear()
        this.group.removeFromParent()
        this.world?.free(); this.world = null; this.ready = null
        this.solids.clear(); this.occupied.clear(); this.binned = new WeakSet()
    }
}
