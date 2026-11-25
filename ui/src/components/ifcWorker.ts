/**
 * THE IFC WORKER (USER 2026-11-25: "no hay manera de ue el IFC se cargue de a partes mientras se muestra la nube?" →
 * "te dije si a esto del IFC"): the IFC is fetched at a LOW priority (behind the cloud's own requests), read by
 * web-ifc here, off the page, and sent in batches of plain arrays — transferred, not copied. The page builds them into
 * the scene a few at a time (IFCLoader.loadIFC) and never stalls.
 *
 * In:  { url }                        — absolute, the page's scan already in it
 * Out: { kind: 'parts', parts }       — a batch of placed geometries (ifcParse.IfcPart), in the order they are read
 *      { kind: 'done', hierarchy }    — the spatial tree, after the last batch
 *      { kind: 'error', message }
 */
import * as WebIFC from 'web-ifc'
import { readIfc, type IfcPart } from './ifcParse'

const scope = self as unknown as {
    postMessage: (message: unknown, transfer?: Transferable[]) => void
    onmessage: ((e: MessageEvent<{ url: string }>) => void) | null
    location: { origin: string }
}

/** a batch is sent once it holds about this many bytes of arrays (a message per element would be thousands) */
const BATCH_BYTES = 8 * 1024 * 1024

scope.onmessage = async e => {
    try {
        const api = new WebIFC.IfcAPI()
        // the page's own copy of web-ifc's engine (public/web-ifc), by its full address: a worker's relative paths
        // are read against the worker's own script
        api.SetWasmPath(new URL('/web-ifc/', scope.location.origin).href, true)
        await api.Init()
        // revalidated every time (an IFC is rewritten in place under its own name; the server answers 304 when it did
        // not change) and LOW priority: the cloud's requests go first
        const response = await fetch(e.data.url, { cache: 'no-cache', priority: 'low' } as RequestInit)
        if (!response.ok) throw new Error(`Failed to fetch IFC: ${response.status}`)
        const modelID = api.OpenModel(new Uint8Array(await response.arrayBuffer()))
        let batch: IfcPart[] = []
        let transfer: Transferable[] = []
        let bytes = 0
        const flush = () => {
            if (!batch.length) return
            scope.postMessage({ kind: 'parts', parts: batch }, transfer)
            batch = []
            transfer = []
            bytes = 0
        }
        const hierarchy = readIfc(api, modelID, part => {
            batch.push(part)
            transfer.push(part.positions.buffer, part.normals.buffer, part.indices.buffer)
            bytes += part.positions.byteLength + part.normals.byteLength + part.indices.byteLength
            if (bytes >= BATCH_BYTES) flush()
        })
        flush()
        api.CloseModel(modelID)
        scope.postMessage({ kind: 'done', hierarchy })
    } catch (err) {
        scope.postMessage({ kind: 'error', message: err instanceof Error ? err.message : String(err) })
    }
}
