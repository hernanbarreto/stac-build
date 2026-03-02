/**
 * IFC Loader — parses IFC files in the browser using web-ifc WASM.
 * 
 * Creates Three.js mesh groups from IFC geometry, preserving
 * the element hierarchy so each IFC element is individually selectable.
 * Also extracts the IFC spatial hierarchy for the BIM Navigator tree.
 */
import * as THREE from 'three'
import * as WebIFC from 'web-ifc'

let ifcApi: WebIFC.IfcAPI | null = null

/** Node in the IFC spatial hierarchy tree */
export interface IFCTreeNode {
    expressID: number
    type: string       // e.g. "IfcProject", "IfcSite", "IfcWall"
    name: string
    children: IFCTreeNode[]
    meshNames: string[] // Three.js mesh names for this element
}

/** Result of loading an IFC file */
export interface IFCLoadResult {
    group: THREE.Group
    hierarchy: IFCTreeNode[]
    filename: string
}

/**
 * Initialize the web-ifc WASM engine (called once).
 */
async function ensureInit(): Promise<WebIFC.IfcAPI> {
    if (ifcApi) return ifcApi

    ifcApi = new WebIFC.IfcAPI()
    ifcApi.SetWasmPath('/web-ifc/')
    await ifcApi.Init()
    console.log('[IFC] ✅ web-ifc WASM initialized')
    return ifcApi
}

/**
 * Extract the length unit scale factor from an IFC model.
 */
function getLengthUnitScale(api: WebIFC.IfcAPI, modelID: number): number {
    try {
        const projectIDs = api.GetLineIDsWithType(modelID, WebIFC.IFCPROJECT)
        if (projectIDs.size() === 0) return 1.0

        const project = api.GetLine(modelID, projectIDs.get(0)) as any
        if (!project?.UnitsInContext) return 1.0

        const unitAssignmentID = project.UnitsInContext.value
        const unitAssignment = api.GetLine(modelID, unitAssignmentID) as any
        if (!unitAssignment?.Units) return 1.0

        for (const unitRef of unitAssignment.Units) {
            const unit = api.GetLine(modelID, unitRef.value) as any
            if (!unit) continue

            const unitType = unit.UnitType?.value
            if (unitType !== 'LENGTHUNIT') continue

            if (unit.type === WebIFC.IFCSIUNIT) {
                const prefix = unit.Prefix?.value
                if (!prefix) return 1.0
                const prefixFactors: Record<string, number> = {
                    'MILLI': 0.001, 'CENTI': 0.01, 'DECI': 0.1, 'KILO': 1000,
                }
                return prefixFactors[prefix] || 1.0
            }

            if (unit.type === WebIFC.IFCCONVERSIONBASEDUNIT) {
                const convFactorID = unit.ConversionFactor?.value
                if (convFactorID) {
                    const convFactor = api.GetLine(modelID, convFactorID) as any
                    if (convFactor?.ValueComponent?.value) {
                        const factor = Number(convFactor.ValueComponent.value)
                        console.log(`[IFC] Length unit: "${unit.Name?.value}" (1 unit = ${factor} m)`)
                        return factor
                    }
                }
            }
        }
    } catch (e) {
        console.warn('[IFC] Could not determine length unit:', e)
    }
    return 1.0
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
function getTypeName(api: WebIFC.IfcAPI, modelID: number, expressID: number): string {
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
function getElementName(api: WebIFC.IfcAPI, modelID: number, expressID: number): string {
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
 * to traverse the tree: Project → Site → Building → Storey → Elements
 */
function buildHierarchy(
    api: WebIFC.IfcAPI,
    modelID: number,
    meshNamesByExpressID: Map<number, string[]>
): IFCTreeNode[] {
    // Build parent→children map from IfcRelAggregates
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

    // Build spatial-structure→contained-elements map from IfcRelContainedInSpatialStructure
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

/**
 * Load an IFC file from a URL and return a Three.js Group
 * plus the spatial hierarchy tree for the BIM Navigator.
 */
export async function loadIFC(url: string, name: string): Promise<IFCLoadResult> {
    const api = await ensureInit()

    const response = await fetch(url)
    if (!response.ok) throw new Error(`Failed to fetch IFC: ${response.status}`)
    const buffer = await response.arrayBuffer()
    const data = new Uint8Array(buffer)

    // console.log(`[IFC] Parsing ${name} (${(data.byteLength / 1024).toFixed(0)} KB)...`)

    const modelID = api.OpenModel(data)
    void getLengthUnitScale(api, modelID)
    // console.log(`[IFC] Unit scale: ...`)

    const allMeshes = api.LoadAllGeometry(modelID)

    const group = new THREE.Group()
    group.name = `ifc_${name}`

    // web-ifc already outputs Y-up coordinates in meters — no transform needed
    const baseTransform = new THREE.Matrix4() // identity

    // Map expressID → mesh names (for hierarchy)
    const meshNamesByExpressID = new Map<number, string[]>()

    for (let i = 0; i < allMeshes.size(); i++) {
        const flatMesh = allMeshes.get(i)
        const expressID = flatMesh.expressID

        const elementName = getElementName(api, modelID, expressID)
        const elementType = getTypeName(api, modelID, expressID)

        const placedGeometries = flatMesh.geometries

        for (let j = 0; j < placedGeometries.size(); j++) {
            const pg = placedGeometries.get(j)

            const geometry = api.GetGeometry(modelID, pg.geometryExpressID)
            const vertexDataPtr = geometry.GetVertexData()
            const vertexDataSize = geometry.GetVertexDataSize()
            const indexDataPtr = geometry.GetIndexData()
            const indexDataSize = geometry.GetIndexDataSize()

            const vertexArray = api.GetVertexArray(vertexDataPtr, vertexDataSize)
            const indexArray = api.GetIndexArray(indexDataPtr, indexDataSize)

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

            const bufferGeometry = new THREE.BufferGeometry()
            bufferGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
            bufferGeometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3))
            bufferGeometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indexArray), 1))

            const color = pg.color
            const r = color.x, g = color.y, b = color.z, a = color.w

            const material = new THREE.MeshStandardMaterial({
                color: new THREE.Color(r, g, b),
                opacity: a,
                transparent: a < 1.0,
                side: THREE.DoubleSide,
                metalness: 0.0,
                roughness: 0.6,
            })

            const mesh = new THREE.Mesh(bufferGeometry, material)

            const placementMatrix = new THREE.Matrix4().fromArray(pg.flatTransformation)
            const combinedTransform = new THREE.Matrix4().multiplyMatrices(baseTransform, placementMatrix)
            mesh.applyMatrix4(combinedTransform)

            const meshName = `${elementType}_${elementName}_${expressID}_${j}`
            mesh.name = meshName
            mesh.userData = {
                expressID,
                ifc_type: elementType,
                ifc_name: elementName,
            }

            group.add(mesh)
            geometry.delete()

            // Track mesh name for hierarchy
            const existing = meshNamesByExpressID.get(expressID) || []
            existing.push(meshName)
            meshNamesByExpressID.set(expressID, existing)
        }
    }

    // Build spatial hierarchy
    const hierarchy = buildHierarchy(api, modelID, meshNamesByExpressID)

    api.CloseModel(modelID)

    console.log(`[IFC] ✅ Loaded ${name}: ${group.children.length} meshes, hierarchy: ${hierarchy.length} roots`)
    return { group, hierarchy, filename: name }
}
