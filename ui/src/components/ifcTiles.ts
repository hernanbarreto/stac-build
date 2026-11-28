/**
 * AN IFC FOR THE VIEWER, AS ITS TILES (USER 2026-11-28: "se pone densa la pagina al mostrar el ifc y ademas aun luego de
 * apagar el ifc queda lento, como lo podemos manejar?, tipo los tiles del glb?" / "si, mejora todo"). The server tiles
 * every IFC of the project (bim/ifc_tiles.py); here the page asks for them — the server starts them when they are
 * missing or of an older version of the file, and the page waits — and reads their elements: the parts' names and
 * colours, the BIM tree. The tiles themselves are streamed by the viewer (Viewport.openIfcTiles) only while shown.
 */
import type { IFCTreeNode } from './ifcParse'

export interface IfcTilesInfo {
    tilesetUrl: string
    parts: Array<{ name: string; expressID: number; ifcType: string; ifcName: string; color: number[] }>
    hierarchy: IFCTreeNode[]
}

const WAIT_MS = 2000

function pause(ms: number, signal?: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
        const id = setTimeout(resolve, ms)
        signal?.addEventListener('abort', () => { clearTimeout(id); reject(new DOMException('IFC load aborted', 'AbortError')) }, { once: true })
    })
}

/**
 * The tiles of the IFC served at `ifcUrl` (`/api/sessions/<id>/bim/<name>`): their tileset's address and the model's
 * elements. `onBuilding` is told once when the server is still making them.
 */
export async function fetchIfcTiles(ifcUrl: string, opts: { signal?: AbortSignal; onBuilding?: () => void } = {}): Promise<IfcTilesInfo> {
    const statusUrl = ifcUrl.replace(/\/bim\/([^/]+)$/, '/bim_tiles/$1/status')
    if (statusUrl === ifcUrl) throw new Error(`not an IFC address: ${ifcUrl}`)
    let told = false
    for (;;) {
        const res = await fetch(statusUrl, { signal: opts.signal, cache: 'no-store' })
        if (!res.ok) throw new Error(`the IFC's tiles: HTTP ${res.status}`)
        const st = await res.json() as { state: string; error?: string; tileset_url?: string; elements_url?: string }
        if (st.state === 'ready' && st.tileset_url && st.elements_url) {
            const el = await fetch(st.elements_url, { signal: opts.signal })
            if (!el.ok) throw new Error(`the IFC's elements: HTTP ${el.status}`)
            const elements = await el.json() as { parts: IfcTilesInfo['parts']; hierarchy: IFCTreeNode[] }
            return { tilesetUrl: st.tileset_url, parts: elements.parts, hierarchy: elements.hierarchy }
        }
        if (st.state === 'failed') throw new Error(st.error || 'the IFC could not be tiled')
        if (st.state !== 'building') throw new Error(`the IFC's tiles: ${st.state}`)
        if (!told) { told = true; opts.onBuilding?.() }
        await pause(WAIT_MS, opts.signal)
    }
}
