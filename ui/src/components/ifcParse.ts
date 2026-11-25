/**
 * THE IFC READ, OFF THE PAGE (USER 2026-11-25: "no hay manera de ue el IFC se cargue de a partes mientras se muestra la
 * nube?, porque no termina de mostrar la nube ni tampoco termina de cargar el IFC" → "te dije si a esto del IFC"):
 * web-ifc's model read element by element — each placed geometry as plain arrays (positions, normals, indices, its
 * placement, its colour), handed over as it is read — and the spatial tree for the BIM navigator. It runs in the IFC
 * worker (ifcWorker.ts); nothing here touches the page or three.js. What it reads is what the page's own loader read
 * before (the same meshes, names and tree), now streamed instead of loaded all at once.
 */
import * as WebIFC from 'web-ifc'

/** Node in the IFC spatial hierarchy tree */
export interface IFCTreeNode {
    expressID: number
    type: string       // e.g. "IfcProject", "IfcSite", "IfcWall"
    name: string
    children: IFCTreeNode[]
    meshNames: string[] // Three.js mesh names for this element
}

/** one placed geometry of an element, as arrays the page builds a mesh from */
export interface IfcPart {
    name: string                 // `${type}_${name}_${expressID}_${j}` — the mesh's name
    expressID: number
    ifcType: string
    ifcName: string
    positions: Float32Array
    normals: Float32Array
    indices: Uint32Array
    matrix: number[]             // its placement, column-major 4x4 (web-ifc's flatTransformation)
    color: [number, number, number, number]
}

// IFC entity type names
const TYPE_NAMES: Record<number, string> = {
    [WebIFC.IFCPROJECT]: 'IfcProject',
    [WebIFC.IFCSITE]: 'IfcSite',
    [WebIFC.IFCBUILDING]: 'IfcBuilding',
    [WebIFC.IFCBUILDINGSTOREY]: 'IfcBuildingStorey',
    [WebIFC.IFCWALL]: 'IfcWall',
    [WebIFC.IFCWALLSTANDARDCASE]: 'IfcWallStandardCase',
    [WebIFC.IFCSLAB]: 'IfcSlab',
    [WebIFC.IFCROOF]: 'IfcRoof',
    [WebIFC.IFCCOLUMN]: 'IfcColumn',
    [WebIFC.IFCBEAM]: 'IfcBeam',
    [WebIFC.IFCDOOR]: 'IfcDoor',
    [WebIFC.IFCWINDOW]: 'IfcWindow',
    [WebIFC.IFCSTAIR]: 'IfcStair',
    [WebIFC.IFCRAILING]: 'IfcRailing',
    [WebIFC.IFCFURNISHINGELEMENT]: 'IfcFurnishingElement',
    [WebIFC.IFCMEMBER]: 'IfcMember',
    [WebIFC.IFCPLATE]: 'IfcPlate',
    [WebIFC.IFCOPENINGELEMENT]: 'IfcOpeningElement',
    [WebIFC.IFCSPACE]: 'IfcSpace',
}

/**
 * Get a human-readable type name for an IFC element.
 */
export function getTypeName(api: WebIFC.IfcAPI, modelID: number, expressID: number): string {
    try {
        const line = api.GetLine(modelID, expressID) as any
        if (!line) return 'Unknown'
        // Check type code against known names
        if (line.type && TYPE_NAMES[line.type]) return TYPE_NAMES[line.type]
        // Fallback: constructor name or type number
        return line.constructor?.name || `Type_${line.type}`
    } catch {
        return 'Unknown'
    }
}

/**
 * Get a human-readable name for an IFC element.
 */
export function getElementName(api: WebIFC.IfcAPI, modelID: number, expressID: number): string {
    try {
        const line = api.GetLine(modelID, expressID) as any
        if (!line) return `Element_${expressID}`
        const name = line.Name?.value
        return name ? String(name) : `Element_${expressID}`
    } catch {
        return `Element_${expressID}`
    }
}

/**
 * Build the spatial hierarchy tree from the IFC model.
 * Uses IfcRelAggregates and IfcRelContainedInSpatialStructure
 * to traverse the tree: Project -> Site -> Building -> Storey -> Elements
 */
export function buildHierarchy(
    api: WebIFC.IfcAPI,
    modelID: number,
    meshNamesByExpressID: Map<number, string[]>
): IFCTreeNode[] {
    // Build parent->children map from IfcRelAggregates
    const childrenMap = new Map<number, number[]>()

    try {
        const relAggIDs = api.GetLineIDsWithType(modelID, WebIFC.IFCRELAGGREGATES)
        for (let i = 0; i < relAggIDs.size(); i++) {
            const rel = api.GetLine(modelID, relAggIDs.get(i)) as any
            if (!rel?.RelatingObject?.value || !rel?.RelatedObjects) continue
            const parentID = rel.RelatingObject.value
            const children = rel.RelatedObjects.map((r: any) => r.value)
            childrenMap.set(parentID, [...(childrenMap.get(parentID) || []), ...children])
        }
    } catch (e) {
        console.warn('[IFC] Could not read IfcRelAggregates:', e)
    }

    // Build spatial-structure->contained-elements map from IfcRelContainedInSpatialStructure
    const containedMap = new Map<number, number[]>()

    try {
        const relContIDs = api.GetLineIDsWithType(modelID, WebIFC.IFCRELCONTAINEDINSPATIALSTRUCTURE)
        for (let i = 0; i < relContIDs.size(); i++) {
            const rel = api.GetLine(modelID, relContIDs.get(i)) as any
            if (!rel?.RelatingStructure?.value || !rel?.RelatedElements) continue
            const structureID = rel.RelatingStructure.value
            const elements = rel.RelatedElements.map((r: any) => r.value)
            containedMap.set(structureID, [...(containedMap.get(structureID) || []), ...elements])
        }
    } catch (e) {
        console.warn('[IFC] Could not read IfcRelContainedInSpatialStructure:', e)
    }

    // Recursive tree builder
    function buildNode(expressID: number): IFCTreeNode {
        const type = getTypeName(api, modelID, expressID)
        const name = getElementName(api, modelID, expressID)
        const meshNames = meshNamesByExpressID.get(expressID) || []

        const childNodes: IFCTreeNode[] = []

        // Aggregated children (spatial children)
        const aggChildren = childrenMap.get(expressID) || []
        for (const childID of aggChildren) {
            childNodes.push(buildNode(childID))
        }

        // Contained elements (building elements inside spatial structures)
        const containedElements = containedMap.get(expressID) || []
        for (const elemID of containedElements) {
            childNodes.push(buildNode(elemID))
        }

        return { expressID, type, name, children: childNodes, meshNames }
    }

    // Find root(s): IfcProject
    const roots: IFCTreeNode[] = []
    try {
        const projectIDs = api.GetLineIDsWithType(modelID, WebIFC.IFCPROJECT)
        for (let i = 0; i < projectIDs.size(); i++) {
            roots.push(buildNode(projectIDs.get(i)))
        }
    } catch (e) {
        console.warn('[IFC] Could not find IfcProject:', e)
    }

    return roots
}

/**
 * Read an open model element by element: every placed geometry with vertices and indices is handed to `onPart` as it
 * is read (with how many of the model's elements have been read, of how many), then the spatial tree is returned.
 */
export function readIfc(
    api: WebIFC.IfcAPI,
    modelID: number,
    onPart: (part: IfcPart, elementsRead: number, elements: number) => void,
): IFCTreeNode[] {
    // Map expressID -> mesh names (for hierarchy)
    const meshNamesByExpressID = new Map<number, string[]>()
    api.StreamAllMeshes(modelID, (flatMesh, index, total) => {
        const expressID = flatMesh.expressID
        const ifcName = getElementName(api, modelID, expressID)
        const ifcType = getTypeName(api, modelID, expressID)
        const placed = flatMesh.geometries
        for (let j = 0; j < placed.size(); j++) {
            const pg = placed.get(j)
            const geometry = api.GetGeometry(modelID, pg.geometryExpressID)
            const vertexArray = api.GetVertexArray(geometry.GetVertexData(), geometry.GetVertexDataSize())
            const indexArray = api.GetIndexArray(geometry.GetIndexData(), geometry.GetIndexDataSize())
            if (vertexArray.length === 0 || indexArray.length === 0) {
                geometry.delete()
                continue
            }
            // web-ifc gives position and normal interleaved, six floats a vertex
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
            const indices = new Uint32Array(indexArray)          // a copy: web-ifc's view is reused by its next call
            geometry.delete()
            const name = `${ifcType}_${ifcName}_${expressID}_${j}`
            const names = meshNamesByExpressID.get(expressID) || []
            names.push(name)
            meshNamesByExpressID.set(expressID, names)
            const c = pg.color
            onPart({ name, expressID, ifcType, ifcName, positions, normals, indices,
                     matrix: Array.from(pg.flatTransformation), color: [c.x, c.y, c.z, c.w] }, index + 1, total)
        }
    })
    return buildHierarchy(api, modelID, meshNamesByExpressID)
}
