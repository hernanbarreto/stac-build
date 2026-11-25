/**
 * IFC Loader — an IFC as a Three.js group of selectable elements, plus its spatial tree for the BIM Navigator.
 *
 * BY PARTS, OFF THE PAGE (USER 2026-11-25: "no hay manera de ue el IFC se cargue de a partes mientras se muestra la
 * nube?, porque no termina de mostrar la nube ni tampoco termina de cargar el IFC, ninguna de las dos" → "te dije si a
 * esto del IFC"): web-ifc reads the file in a worker (ifcWorker.ts — fetched at a low priority, behind the cloud) and
 * sends its elements in batches; here they are built into the model's group a few milliseconds per frame, so the page
 * — and the cloud streaming in it — never stalls, and a group already in the scene fills in as its parts arrive.
 * The meshes, their names, materials, placements and the tree are the ones the page's own loader built before.
 */
import * as THREE from 'three'
import { withScan } from '../net/scanHeader'
import type { IFCTreeNode, IfcPart } from './ifcParse'

export type { IFCTreeNode } from './ifcParse'

/** Result of loading an IFC file */
export interface IFCLoadResult {
    group: THREE.Group
    hierarchy: IFCTreeNode[]
    filename: string
}

export interface IFCLoadOptions {
    /** the model's group as soon as it exists (still empty): put in the scene, its elements appear as they are built */
    onGroup?: (group: THREE.Group) => void
    /** an abort stops the worker and the building; the promise rejects with an AbortError */
    signal?: AbortSignal
}

/** The opacity every BIM element is loaded with, and where the BIM panel's sliders start (USER 2026-11-19: "en el
 *  apartado BIM todos los elementos ademas del arbol general deben tener la opcion de slide de opacidad y por
 *  defecto estar al 80%"). */
export const BIM_DEFAULT_OPACITY = 0.8

/** the time, each frame, the page spends turning received parts into meshes (the rest of the frame is the scene's) */
const FRAME_BUDGET_MS = 8

/**
 * Count total descendant elements (including self) in a hierarchy node.
 */
export function countDescendants(node: IFCTreeNode): number {
    let count = 1
    for (const child of node.children) {
        count += countDescendants(child)
    }
    return count
}

/**
 * Collect all mesh names from a hierarchy node and its descendants.
 */
export function collectMeshNames(node: IFCTreeNode): string[] {
    const names = [...node.meshNames]
    for (const child of node.children) {
        names.push(...collectMeshNames(child))
    }
    return names
}

/** one received part as the mesh the page's loader made of it: its own material, its placement applied */
function buildMesh(part: IfcPart): THREE.Mesh {
    const bufferGeometry = new THREE.BufferGeometry()
    bufferGeometry.setAttribute('position', new THREE.BufferAttribute(part.positions, 3))
    bufferGeometry.setAttribute('normal', new THREE.BufferAttribute(part.normals, 3))
    bufferGeometry.setIndex(new THREE.BufferAttribute(part.indices, 1))
    const [r, g, b, a] = part.color
    const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, g, b),
        opacity: Math.min(a, BIM_DEFAULT_OPACITY),      // (a material the IFC makes more transparent keeps its own)
        transparent: Math.min(a, BIM_DEFAULT_OPACITY) < 1.0,
        side: THREE.DoubleSide,
        metalness: 0.0,
        roughness: 0.6,
    })
    const mesh = new THREE.Mesh(bufferGeometry, material)
    // web-ifc already outputs Y-up coordinates in meters: its placement alone
    mesh.applyMatrix4(new THREE.Matrix4().fromArray(part.matrix))
    mesh.name = part.name
    mesh.userData = {
        expressID: part.expressID,
        ifc_type: part.ifcType,
        ifc_name: part.ifcName,
    }
    return mesh
}

/**
 * Load an IFC file from a URL: its Three.js group (handed to `onGroup` at once, filled by parts) and, when the last
 * part is built, the spatial hierarchy tree for the BIM Navigator.
 */
export function loadIFC(url: string, name: string, opts: IFCLoadOptions = {}): Promise<IFCLoadResult> {
    const group = new THREE.Group()
    group.name = `ifc_${name}`
    return new Promise<IFCLoadResult>((resolve, reject) => {
        if (opts.signal?.aborted) {
            reject(new DOMException('IFC load aborted', 'AbortError'))
            return
        }
        opts.onGroup?.(group)
        const worker = new Worker(new URL('./ifcWorker.ts', import.meta.url), { type: 'module' })
        const queue: IfcPart[] = []
        let head = 0                                     // the next part of `queue` to build
        let hierarchy: IFCTreeNode[] | null = null
        let scheduled = false
        let finished = false
        const finish = (err?: Error) => {
            if (finished) return
            finished = true
            worker.terminate()
            opts.signal?.removeEventListener('abort', onAbort)
            if (err) {
                reject(err)
                return
            }
            console.log(`[IFC] Loaded ${name}: ${group.children.length} meshes, hierarchy: ${hierarchy?.length ?? 0} roots`)
            resolve({ group, hierarchy: hierarchy ?? [], filename: name })
        }
        const build = () => {
            scheduled = false
            if (finished) return
            const t0 = performance.now()
            while (head < queue.length && performance.now() - t0 < FRAME_BUDGET_MS) {
                group.add(buildMesh(queue[head]))
                queue[head++] = undefined as unknown as IfcPart      // its arrays now belong to the mesh alone
            }
            if (head < queue.length) schedule()
            else {
                queue.length = 0
                head = 0
                if (hierarchy) finish()
            }
        }
        const schedule = () => {
            if (scheduled || finished) return
            scheduled = true
            requestAnimationFrame(build)
        }
        const onAbort = () => finish(new DOMException('IFC load aborted', 'AbortError'))
        opts.signal?.addEventListener('abort', onAbort)
        worker.onmessage = (e: MessageEvent<{ kind: string; parts?: IfcPart[]; hierarchy?: IFCTreeNode[]; message?: string }>) => {
            const m = e.data
            if (m.kind === 'parts' && m.parts) {
                for (const part of m.parts) queue.push(part)      // (a batch can hold tens of thousands of small parts)
                schedule()
            } else if (m.kind === 'done') {
                hierarchy = m.hierarchy ?? []
                schedule()
            } else if (m.kind === 'error') {
                finish(new Error(m.message || 'the IFC could not be read'))
            }
        }
        worker.onerror = e => finish(new Error(e.message || 'the IFC worker failed'))
        // the page's scan goes in the address (the worker's requests carry no page header), the address made whole
        worker.postMessage({ url: new URL(withScan(url), location.href).href })
    })
}
