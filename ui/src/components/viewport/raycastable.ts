/**
 * WHAT A RAY OF THE VIEWER MAY TOUCH, AND HOW FAST (USER 2026-11-28: "se pone densa la pagina al mostrar el ifc y ademas
 * aun luego de apagar el ifc queda lento" / "debe moverse fluido siempre" / "tanto para puntos, ifc y glb, para todo").
 *
 * three.js casts a ray at INVISIBLE objects too, and tests every triangle of a mesh whose bounding sphere it crosses:
 * the BIM hover (every 80 ms of mouse movement) cast at every triangle of every IFC of the session, ticked or not — on
 * grupo_electrogeno's IFC (7.54 M triangles) up to 778 ms a ray (measured 2026-11-28, scratchpad ifc_ray.cjs); with a
 * bounding-volume tree (three-mesh-bvh) 6.5 ms at most.
 *  · visibleMeshes(): the meshes a ray may touch — only those shown, the whole chain of parents shown.
 *  · queueBounds(): a mesh's tree, built while the page is idle (one mesh at a time, never in the middle of a frame's
 *    work); until it has one, `hasBounds()` is false and the hover leaves it out (a click still casts at it).
 */
import * as THREE from 'three'
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree, CENTER } from 'three-mesh-bvh'

type BvhGeometry = THREE.BufferGeometry & {
    boundsTree?: unknown
    computeBoundsTree?: typeof computeBoundsTree
    disposeBoundsTree?: typeof disposeBoundsTree
}

// the tree's ray for every mesh (a mesh with no tree falls back to three's own)
;(THREE.BufferGeometry.prototype as BvhGeometry).computeBoundsTree = computeBoundsTree
;(THREE.BufferGeometry.prototype as BvhGeometry).disposeBoundsTree = disposeBoundsTree
THREE.Mesh.prototype.raycast = acceleratedRaycast

/** the meshes under these roots that are on screen (each one and all its parents visible) */
export function visibleMeshes(roots: Array<THREE.Object3D | null | undefined>): THREE.Mesh[] {
    const out: THREE.Mesh[] = []
    for (const root of roots) {
        if (!root) continue
        let shown = true
        for (let p: THREE.Object3D | null = root; p; p = p.parent) if (!p.visible) { shown = false; break }
        if (!shown) continue
        root.traverseVisible(o => { if ((o as THREE.Mesh).isMesh) out.push(o as THREE.Mesh) })
    }
    return out
}

export function hasBounds(mesh: THREE.Mesh): boolean {
    return !!(mesh.geometry as BvhGeometry).boundsTree
}

const queue: THREE.Mesh[] = []
let scheduled = false

const idle: (cb: (d: { timeRemaining: () => number }) => void) => void =
    typeof window !== 'undefined' && 'requestIdleCallback' in window
        ? cb => (window as unknown as { requestIdleCallback: (f: typeof cb, o: object) => void }).requestIdleCallback(cb, { timeout: 1000 })
        : cb => setTimeout(() => cb({ timeRemaining: () => 8 }), 50)

function work(deadline: { timeRemaining: () => number }) {
    scheduled = false
    while (queue.length && deadline.timeRemaining() > 2) {
        const mesh = queue.shift()!
        const geo = mesh.geometry as BvhGeometry
        // a mesh that left the scene, or was disposed, while it waited
        if (geo.boundsTree || !mesh.parent || !geo.getAttribute('position')) continue
        geo.computeBoundsTree!({ strategy: CENTER, targetLeafSize: 10 })
    }
    if (queue.length) schedule()
}

function schedule() {
    if (scheduled) return
    scheduled = true
    idle(work)
}

/** these meshes' trees, built when the page is idle */
export function queueBounds(meshes: Iterable<THREE.Mesh>): void {
    for (const m of meshes) {
        const geo = m.geometry as BvhGeometry
        if (!geo.boundsTree && geo.getAttribute('position')) queue.push(m)
    }
    if (queue.length) schedule()
}

/** a mesh's tree freed with its geometry */
export function disposeBounds(mesh: THREE.Mesh): void {
    const geo = mesh.geometry as BvhGeometry
    if (geo.boundsTree) geo.disposeBoundsTree!()
}
