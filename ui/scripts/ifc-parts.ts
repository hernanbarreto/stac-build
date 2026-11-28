/**
 * AN IFC'S ELEMENTS FOR ITS TILES (USER 2026-11-28: "se pone densa la pagina al mostrar el ifc y ademas aun luego de
 * apagar el ifc queda lento, como lo podemos manejar?, tipo los tiles del glb?" / "las soluciones siempre deben ser
 * completas y de fondo"). The IFC read by the page's own reader (ifcParse.readIfc — the same parts, names, colours and
 * spatial tree the viewer builds), each part's vertices placed in the model's frame, written for the tiler
 * (tools/glb/ifc_tiles.mjs):
 *   <out>.json  { parts: [{ name, expressID, ifcType, ifcName, color, vertices, indices, offset }], hierarchy }
 *   <out>.bin   per part: positions (float32 x3), normals (float32 x3), indices (uint32), at its offset
 *
 *   npx esbuild scripts/ifc-parts.ts --bundle --platform=node --format=cjs --external:web-ifc --outfile=<bundle.cjs>
 *   NODE_PATH=node_modules node <bundle.cjs> <file.ifc> <out>
 */
import * as fs from 'node:fs'
import * as WebIFC from 'web-ifc'
import { readIfc, type IfcPart } from '../src/components/ifcParse'

/** a part's vertices and normals in the model's frame: its column-major placement applied (three's applyMatrix4) */
function place(part: IfcPart): { pos: Float32Array; nrm: Float32Array } {
    const m = part.matrix, p = part.positions, n = part.normals
    const pos = new Float32Array(p.length), nrm = new Float32Array(n.length)
    // the normal matrix: the inverse transpose of the upper 3x3
    const a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9], i = m[10]
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g
    const det = a * A + b * B + c * C
    const N = [A / det, B / det, C / det, -(b * i - c * h) / det, (a * i - c * g) / det, -(a * h - b * g) / det,
               (b * f - c * e) / det, -(a * f - c * d) / det, (a * e - b * d) / det]
    for (let v = 0; v < p.length; v += 3) {
        const x = p[v], y = p[v + 1], z = p[v + 2]
        pos[v] = m[0] * x + m[4] * y + m[8] * z + m[12]
        pos[v + 1] = m[1] * x + m[5] * y + m[9] * z + m[13]
        pos[v + 2] = m[2] * x + m[6] * y + m[10] * z + m[14]
        const q = n[v], r = n[v + 1], s = n[v + 2]
        const nx = N[0] * q + N[3] * r + N[6] * s, ny = N[1] * q + N[4] * r + N[7] * s, nz = N[2] * q + N[5] * r + N[8] * s
        const l = Math.hypot(nx, ny, nz) || 1
        nrm[v] = nx / l; nrm[v + 1] = ny / l; nrm[v + 2] = nz / l
    }
    return { pos, nrm }
}

async function main() {
    const [file, out] = process.argv.slice(2)
    if (!file || !out) {
        console.error('usage: node <bundle.cjs> <file.ifc> <out prefix>')
        process.exit(2)
    }
    const t0 = Date.now()
    const api = new WebIFC.IfcAPI()
    await api.Init()
    const modelID = api.OpenModel(new Uint8Array(fs.readFileSync(file)))
    if (modelID < 0) throw new Error(`${file}: web-ifc could not open it`)
    const fd = fs.openSync(`${out}.bin.part`, 'w')
    const parts: object[] = []
    let offset = 0
    const write = (arr: Float32Array | Uint32Array) => {
        const buf = Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength)
        fs.writeSync(fd, buf)
        offset += buf.byteLength
    }
    const hierarchy = readIfc(api, modelID, part => {
        const { pos, nrm } = place(part)
        parts.push({ name: part.name, expressID: part.expressID, ifcType: part.ifcType, ifcName: part.ifcName,
                     color: part.color, vertices: pos.length / 3, indices: part.indices.length, offset })
        write(pos); write(nrm); write(part.indices)
    })
    fs.closeSync(fd)
    api.CloseModel(modelID)
    fs.writeFileSync(`${out}.json.part`, JSON.stringify({ source: file, parts, hierarchy }))
    fs.renameSync(`${out}.bin.part`, `${out}.bin`)
    fs.renameSync(`${out}.json.part`, `${out}.json`)
    console.log(`[ifc-parts] ${file}: ${parts.length} parts, ${(offset / 1e6).toFixed(0)} MB (${((Date.now() - t0) / 1000).toFixed(1)} s)`)
}

main().catch(err => {
    console.error('[ifc-parts] FAILED:', err?.stack || err)
    process.exit(1)
})
