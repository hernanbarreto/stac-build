/**
 * THE IFC READ BY PARTS GIVES WHAT THE PAGE'S LOADER BUILT BEFORE (USER 2026-11-25: "te dije si a esto del IFC").
 * ifcParse.readIfc — streamed element by element in the worker — against the page loader's former path (web-ifc's
 * LoadAllGeometry, all at once), on a real IFC: the same parts (names), each with the same positions, normals,
 * indices, placement and colour, bit for bit, and the same spatial tree.
 *
 *   npx esbuild scripts/check-ifc-read.ts --bundle --platform=node --format=cjs --external:web-ifc --outfile=<out.cjs>
 *   NODE_PATH=node_modules node <out.cjs> <file.ifc> [...]
 */
import * as fs from 'node:fs'
import * as WebIFC from 'web-ifc'
import { buildHierarchy, getElementName, getTypeName, readIfc, type IfcPart } from '../src/components/ifcParse'

interface Ref { positions: Float32Array; normals: Float32Array; indices: Uint32Array; matrix: number[]; color: number[] }

/** the page loader's former path, as it was (IFCLoader.ts before the worker) */
function formerRead(api: WebIFC.IfcAPI, modelID: number) {
    const parts = new Map<string, Ref>()
    const meshNamesByExpressID = new Map<number, string[]>()
    const allMeshes = api.LoadAllGeometry(modelID)
    for (let i = 0; i < allMeshes.size(); i++) {
        const flatMesh = allMeshes.get(i)
        const expressID = flatMesh.expressID
        const elementName = getElementName(api, modelID, expressID)
        const elementType = getTypeName(api, modelID, expressID)
        const placedGeometries = flatMesh.geometries
        for (let j = 0; j < placedGeometries.size(); j++) {
            const pg = placedGeometries.get(j)
            const geometry = api.GetGeometry(modelID, pg.geometryExpressID)
            const vertexArray = api.GetVertexArray(geometry.GetVertexData(), geometry.GetVertexDataSize())
            const indexArray = api.GetIndexArray(geometry.GetIndexData(), geometry.GetIndexDataSize())
            if (vertexArray.length === 0 || indexArray.length === 0) {
                geometry.delete()
                continue
            }
            const numVertices = vertexArray.length / 6
            const positions = new Float32Array(numVertices * 3)
            const normals = new Float32Array(numVertices * 3)
            for (let v = 0; v < numVertices; v++) {
                positions[v * 3] = vertexArray[v * 6]
                positions[v * 3 + 1] = vertexArray[v * 6 + 1]
                positions[v * 3 + 2] = vertexArray[v * 6 + 2]
                normals[v * 3] = vertexArray[v * 6 + 3]
                normals[v * 3 + 1] = vertexArray[v * 6 + 4]
                normals[v * 3 + 2] = vertexArray[v * 6 + 5]
            }
            const name = `${elementType}_${elementName}_${expressID}_${j}`
            parts.set(name, { positions, normals, indices: new Uint32Array(indexArray),
                              matrix: Array.from(pg.flatTransformation), color: [pg.color.x, pg.color.y, pg.color.z, pg.color.w] })
            geometry.delete()
            const existing = meshNamesByExpressID.get(expressID) || []
            existing.push(name)
            meshNamesByExpressID.set(expressID, existing)
        }
    }
    return { parts, hierarchy: buildHierarchy(api, modelID, meshNamesByExpressID) }
}

const same = (a: ArrayLike<number>, b: ArrayLike<number>) => {
    if (a.length !== b.length) return false
    for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false
    return true
}

async function main() {
    let failed = 0
    for (const file of process.argv.slice(2)) {
        const data = new Uint8Array(fs.readFileSync(file))
        const api = new WebIFC.IfcAPI()
        await api.Init()
        let t = Date.now()
        const m1 = api.OpenModel(data)
        const former = formerRead(api, m1)
        api.CloseModel(m1)
        const tFormer = (Date.now() - t) / 1000
        t = Date.now()
        const m2 = api.OpenModel(data)
        const streamed: IfcPart[] = []
        let calls = 0, lastRead = 0, elements = 0
        const hierarchy = readIfc(api, m2, (part, read, total) => { streamed.push(part); calls++; lastRead = read; elements = total })
        api.CloseModel(m2)
        const tStreamed = (Date.now() - t) / 1000
        const problems: string[] = []
        if (streamed.length !== former.parts.size) problems.push(`parts ${streamed.length} streamed, ${former.parts.size} former`)
        const seen = new Set<string>()
        for (const p of streamed) {
            if (seen.has(p.name)) problems.push(`part twice: ${p.name}`)
            seen.add(p.name)
            const r = former.parts.get(p.name)
            if (!r) { problems.push(`part not in the former read: ${p.name}`); continue }
            if (!same(p.positions, r.positions)) problems.push(`positions differ: ${p.name}`)
            if (!same(p.normals, r.normals)) problems.push(`normals differ: ${p.name}`)
            if (!same(p.indices, r.indices)) problems.push(`indices differ: ${p.name}`)
            if (!same(p.matrix, r.matrix)) problems.push(`placement differs: ${p.name}`)
            if (!same(p.color, r.color)) problems.push(`colour differs: ${p.name}`)
        }
        if (JSON.stringify(hierarchy) !== JSON.stringify(former.hierarchy)) problems.push('the spatial tree differs')
        if (lastRead !== elements) problems.push(`the progress ended at ${lastRead} of ${elements} elements`)
        const name = file.split('/').pop()
        console.log(`${name}: ${streamed.length} parts in ${calls} hand-overs, ${elements} elements; former ${tFormer.toFixed(1)} s, ` +
                    `streamed ${tStreamed.toFixed(1)} s — ${problems.length ? `${problems.length} DIFFERENCES` : 'identical'}`)
        for (const q of problems.slice(0, 8)) console.log('   ', q)
        failed += problems.length ? 1 : 0
    }
    console.log(failed ? 'NOT IDENTICAL' : 'IDENTICAL')
    process.exit(failed ? 1 : 0)
}

void main()
