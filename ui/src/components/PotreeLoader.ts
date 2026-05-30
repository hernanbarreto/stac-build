/**
 * STAC Build — Potree 2.0 Octree Loader
 * Lightweight LOD point cloud loader for Three.js
 * 
 * Loads Potree 2.0 format (PotreeConverter 2.1 output):
 *   metadata.json — bounding box, attributes, hierarchy info
 *   hierarchy.bin — octree node index (which nodes exist + byte offsets)
 *   octree.bin    — actual point data per node
 * 
 * Features:
 *   - Frustum culling (only loads visible nodes)
 *   - Distance-based LOD (more detail near camera)
 *   - Point budget (never exceed N points on screen)
 *   - Dynamic loading/unloading as camera moves
 * 
 * Hernán Barreto — Ingerop IN3 Session IV
 */

import * as THREE from 'three'

// ── Types ──────────────────────────────────────────────────────────────

interface PotreeMetadata {
    version: string
    name: string
    points: number
    hierarchy: {
        firstChunkSize: number
        stepSize: number
        depth: number
    }
    offset: [number, number, number]
    scale: [number, number, number]
    spacing: number
    boundingBox: {
        min: [number, number, number]
        max: [number, number, number]
    }
    encoding: string
    attributes: PotreeAttribute[]
}

interface PotreeAttribute {
    name: string
    size: number
    numElements: number
    elementSize: number
    type: string
    min?: number[]
    max?: number[]
    scale?: number[]
    offset?: number[]
}

interface OctreeNode {
    name: string
    level: number
    childMask: number
    numPoints: number
    byteOffset: bigint
    byteSize: bigint
    boundingBox: THREE.Box3
    loaded: boolean
    loading: boolean
    points: THREE.Points | null
    nodeType: number  // 0=normal, 1=leaf, 2=proxy (hierarchy chunk)
    hierarchyByteOffset: bigint  // for proxy nodes: offset in hierarchy.bin
    hierarchyByteSize: bigint    // for proxy nodes: size in hierarchy.bin
    hierarchyLoaded: boolean     // for proxy nodes: has chunk been loaded?
    hierarchyLoading: boolean    // for proxy nodes: is chunk being loaded?
    lastUsed: number             // LRU tick: last time this node was shown
}

// ── Potree Octree Loader ───────────────────────────────────────────────

export class PotreeOctreeLoader {
    private baseUrl: string
    private metadata: PotreeMetadata | null = null
    private nodes: Map<string, OctreeNode> = new Map()
    private scene: THREE.Scene
    private camera: THREE.PerspectiveCamera
    private material: THREE.ShaderMaterial
    private pointBudget: number
    private visiblePoints = 0
    private octreeGroup: THREE.Group
    private hierarchyData: ArrayBuffer | null = null
    private totalLoadedPoints = 0
    // Aborts all in-flight fetches (metadata/hierarchy/per-node octree ranges)
    // when this loader is disposed. Without it, abandoned fetches from a previous
    // session pile up in the browser's connection pool (cap ~6/host) and starve
    // the next load → progressively slower, then nothing (only F5 cleared it).
    private _abort = new AbortController()
    // In-memory node cache: nodes leaving the frustum keep their parsed geometry
    // (just removed from the scene) so re-entering is instant — no HTTP re-fetch.
    // Bounded by an LRU cap to keep client RAM in check.
    private _cachedPoints = 0
    private _tick = 0
    private _loading = false
    private _smallCloudLogged = false
    private _hasConfidence = false
    private _forceClassId: number | undefined = undefined

    constructor(
        scene: THREE.Scene,
        camera: THREE.PerspectiveCamera,
        material: THREE.ShaderMaterial,
        pointBudget = 5_000_000,
        forceClassId?: number,
    ) {
        this.baseUrl = ''
        this.scene = scene
        this.camera = camera
        this.material = material
        this.pointBudget = pointBudget
        this._forceClassId = forceClassId
        this.octreeGroup = new THREE.Group()
        this.octreeGroup.name = 'potree-octree'
        this.scene.add(this.octreeGroup)
    }

    get loading(): boolean { return this._loading }
    get pointCount(): number { return this.totalLoadedPoints }
    get isLoaded(): boolean { return this.metadata !== null }

    /** Apply a 4x4 transform matrix (column-major float array) to the octree group */
    setTransform(matrix4: number[]): void {
        if (matrix4.length !== 16) return
        const m = new THREE.Matrix4()
        m.fromArray(matrix4)
        // Decompose and apply to position/quaternion/scale so that
        // matrixAutoUpdate keeps working for newly added child nodes
        const pos = new THREE.Vector3()
        const quat = new THREE.Quaternion()
        const scl = new THREE.Vector3()
        m.decompose(pos, quat, scl)
        this.octreeGroup.position.copy(pos)
        this.octreeGroup.quaternion.copy(quat)
        this.octreeGroup.scale.copy(scl)
        this.octreeGroup.updateMatrixWorld(true)
        // console.log('[PotreeLoader] Applied floor alignment transform')
    }

    /** Get the octree Group for direct matrix manipulation (alignment tool) */
    getOctreeGroup(): THREE.Group {
        return this.octreeGroup
    }

    setPointBudget(budget: number): void {
        this.pointBudget = budget
    }

    /** Load a Potree 2.0 dataset from a base URL */
    async load(baseUrl: string): Promise<number> {
        this._loading = true
        this.baseUrl = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/'

        try {
            // 1. Load metadata (no-store: corrected cloud may replace original)
            const metaResp = await fetch(this.baseUrl + 'metadata.json', { cache: 'no-store', signal: this._abort.signal })
            if (!metaResp.ok) throw new Error(`Failed to load metadata: ${metaResp.status}`)
            this.metadata = await metaResp.json()

            // console.log(`[PotreeLoader] Loaded metadata: ${this.metadata!.points.toLocaleString()} points, depth ${this.metadata!.hierarchy.depth}`)

            // 2. Load hierarchy (node index)
            const hierResp = await fetch(this.baseUrl + 'hierarchy.bin', { cache: 'no-store', signal: this._abort.signal })
            if (!hierResp.ok) throw new Error(`Failed to load hierarchy: ${hierResp.status}`)
            this.hierarchyData = await hierResp.arrayBuffer()

            // 3. Parse hierarchy to build node tree
            this.parseHierarchy()

            // 4. Octree point data is streamed PER-NODE via HTTP Range requests
            //    in loadNode() — we never download the whole octree.bin (can be
            //    multiple GB). Only visible nodes (within the point budget) fetch
            //    their byte range. This is what makes large clouds load at all.

            // 5. Initial visibility update — fetches root + nearby nodes
            this.updateVisibility()

            this._loading = false
            return this.totalLoadedPoints
        } catch (e) {
            this._loading = false
            throw e
        }
    }

    /** Parse hierarchy.bin (or a sub-chunk) into OctreeNode map.
     *  Potree 2.0 hierarchy is chunked: type=2 nodes are proxies whose
     *  byteOffset/byteSize point to a sub-range of hierarchy.bin that
     *  must be fetched separately to expand the tree deeper.
     */
    private parseHierarchyChunk(
        buffer: ArrayBuffer,
        bufferOffset: number,
        bufferSize: number,
        rootName: string
    ): void {
        if (!this.metadata) return

        const meta = this.metadata
        const data = new DataView(buffer, bufferOffset, bufferSize)
        const bboxMin = new THREE.Vector3(...meta.boundingBox.min)
        const boxSize = new THREE.Vector3(...meta.boundingBox.max).sub(bboxMin)

        const entrySize = 22
        const numEntries = Math.floor(bufferSize / entrySize)

        // BFS: start with the root of this chunk
        const nodeQueue: string[] = [rootName]
        let entryIdx = 0

        while (entryIdx < numEntries && entryIdx < nodeQueue.length) {
            const name = nodeQueue[entryIdx]
            const off = entryIdx * entrySize

            if (off + entrySize > bufferSize) break

            const nodeType = data.getUint8(off + 0)
            const childMask = data.getUint8(off + 1)
            const numPoints = data.getUint32(off + 2, true)
            const byteOffset = data.getBigUint64(off + 6, true)
            const byteSize = data.getBigUint64(off + 14, true)

            const nodeBox = this.computeNodeBoundingBox(name, bboxMin, boxSize)

            // Check if this node already exists (proxy being expanded)
            const existing = this.nodes.get(name)
            if (existing && existing.nodeType === 2) {
                // Replace proxy with real node
                existing.nodeType = nodeType
                existing.byteOffset = byteOffset
                existing.byteSize = byteSize
                existing.numPoints = numPoints
                existing.childMask = childMask
                existing.hierarchyLoaded = true
                existing.hierarchyLoading = false
            } else if (!existing) {
                const node: OctreeNode = {
                    name,
                    level: name.length - 1,
                    childMask,
                    numPoints,
                    byteOffset,
                    byteSize,
                    boundingBox: nodeBox,
                    loaded: false,
                    loading: false,
                    points: null,
                    nodeType,
                    hierarchyByteOffset: nodeType === 2 ? byteOffset : 0n,
                    hierarchyByteSize: nodeType === 2 ? byteSize : 0n,
                    hierarchyLoaded: nodeType !== 2,
                    hierarchyLoading: false,
                    lastUsed: 0,
                }
                this.nodes.set(name, node)
            }

            // Type 2 (proxy): don't expand children from this chunk
            // Children will be available after loading the proxy's hierarchy chunk
            if (nodeType !== 2) {
                for (let c = 0; c < 8; c++) {
                    if (childMask & (1 << c)) {
                        nodeQueue.push(name + c)
                    }
                }
            }

            entryIdx++
        }
    }

    /** Initial hierarchy parse from the full hierarchy.bin */
    private parseHierarchy(): void {
        if (!this.hierarchyData || !this.metadata) return

        const firstChunkSize = this.metadata.hierarchy.firstChunkSize
        const parseSize = Math.min(firstChunkSize, this.hierarchyData.byteLength)
        // console.log(`[PotreeLoader] Parsing hierarchy: firstChunkSize=${firstChunkSize}, totalHierarchySize=${this.hierarchyData.byteLength}, entries=${Math.floor(parseSize / 22)}`)
        this.parseHierarchyChunk(this.hierarchyData, 0, parseSize, 'r')

        // Detailed stats per level
        const levelStats = new Map<number, { count: number; points: number; proxies: number }>()
        for (const [, n] of this.nodes) {
            const s = levelStats.get(n.level) || { count: 0, points: 0, proxies: 0 }
            s.count++
            s.points += n.numPoints
            if (n.nodeType === 2) s.proxies++
            levelStats.set(n.level, s)
        }
        // console.log(`[PotreeLoader] Hierarchy stats:`)
        // for (const [level, s] of Array.from(levelStats.entries()).sort((a, b) => a[0] - b[0])) {
        //     console.log(`  Level ${level}: ${s.count} nodes, ${s.points.toLocaleString()} pts${s.proxies > 0 ? `, ${s.proxies} proxies` : ''}`)
        // }
        // console.log(`[PotreeLoader] Total: ${this.nodes.size} nodes`)
    }

    /** Load a proxy node's hierarchy chunk via HTTP Range request */
    private async loadProxyHierarchy(node: OctreeNode): Promise<void> {
        if (!this.hierarchyData || node.hierarchyLoaded || node.hierarchyLoading) return
        node.hierarchyLoading = true

        try {
            const offset = Number(node.hierarchyByteOffset)
            const size = Number(node.hierarchyByteSize)

            if (offset + size <= this.hierarchyData.byteLength) {
                // Sub-chunk is within our already-loaded hierarchy.bin
                this.parseHierarchyChunk(this.hierarchyData, offset, size, node.name)
            } else {
                // Need to fetch via Range request
                const resp = await fetch(this.baseUrl + 'hierarchy.bin', {
                    headers: { 'Range': `bytes=${offset}-${offset + size - 1}` },
                    signal: this._abort.signal,
                })
                const buffer = await resp.arrayBuffer()
                this.parseHierarchyChunk(buffer, 0, buffer.byteLength, node.name)
            }

            node.hierarchyLoaded = true
            node.hierarchyLoading = false

            // Debug: proxy count
            // const proxyCount = Array.from(this.nodes.values()).filter(n => n.nodeType === 2 && !n.hierarchyLoaded).length
            // console.log(`[PotreeLoader] Expanded proxy '${node.name}' → ${this.nodes.size} total nodes (${proxyCount} proxies remaining)`)
        } catch (e) {
            node.hierarchyLoading = false
            console.error(`[PotreeLoader] Failed to load hierarchy for proxy '${node.name}':`, e)
        }
    }

    /** Compute bounding box for a node given its name path */
    private computeNodeBoundingBox(name: string, rootMin: THREE.Vector3, rootSize: THREE.Vector3): THREE.Box3 {
        const min = rootMin.clone()
        const size = rootSize.clone()

        // Skip 'r' prefix, then each digit determines which octant
        for (let i = 1; i < name.length; i++) {
            const childIdx = parseInt(name[i])
            size.multiplyScalar(0.5)

            // Octant index in Potree 2: bit0=Z, bit1=Y, bit2=X
            if (childIdx & 1) min.z += size.z
            if (childIdx & 2) min.y += size.y
            if (childIdx & 4) min.x += size.x
        }

        return new THREE.Box3(min, min.clone().add(size))
    }

    /** Update visibility — Potree-style priority queue traversal.
     *  Based on the real Potree updateVisibility algorithm:
     *  1. Start with root at MAX priority
     *  2. Pop highest-priority node → SHOW it (additive)
     *  3. Add its children to queue IF their projected pixel size >= threshold
     *  4. Stop when budget is reached
     *  Parents are always shown BEFORE children, ensuring base coverage.
     *
     *  OPTIMIZATION: For small clouds that fit entirely within the point budget,
     *  we load ALL nodes and skip LOD culling. This prevents the "disappearing
     *  chunks" issue that occurs when the LOD algorithm unloads nodes during
     *  camera movement. A regular PLY viewer shows all points all the time —
     *  for small clouds, we do the same.
     */
    updateVisibility(): void {
        if (!this.metadata) return

        // ── Small cloud bypass ─────────────────────────────────
        // If the cloud has fewer points than the budget, load EVERYTHING
        // and skip frustum culling/unloading entirely. This matches the
        // behavior of a standard viewer while keeping LOD for large clouds.
        if (this.metadata.points <= this.pointBudget) {
            this._updateVisibilityLoadAll()
            if (!this._smallCloudLogged) {
                console.log(`[PotreeLoader] ✅ Small cloud bypass: ${this.metadata.points.toLocaleString()} pts <= budget ${this.pointBudget.toLocaleString()}, loading ALL nodes (${this.nodes.size} nodes)`)
                this._smallCloudLogged = true
            }
            return
        }

        const frustum = new THREE.Frustum()
        const projScreenMatrix = new THREE.Matrix4()
        // Include the octreeGroup's world matrix so local-space bounding boxes
        // are correctly tested against the camera's frustum
        const groupWorldMatrix = this.octreeGroup.matrixWorld
        projScreenMatrix.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse)
        projScreenMatrix.multiply(groupWorldMatrix)
        frustum.setFromProjectionMatrix(projScreenMatrix)

        // Transform camera position to local space for distance calculations
        const cameraPos = this.camera.position.clone()
        const groupWorldInverse = groupWorldMatrix.clone().invert()
        cameraPos.applyMatrix4(groupWorldInverse)

        const domHeight = window.innerHeight || 800
        const fovRad = (this.camera.fov * Math.PI) / 180
        const slope = Math.tan(fovRad / 2)

        // Minimum projected pixel size for a child to be queued (Potree default: 15-30)
        // Set to 15 to ensure the deepest LOD nodes load properly when zoomed in.
        const minimumNodePixelSize = 15;

        // Priority queue: [weight, nodeName] — process highest weight first
        // Using simple sorted array (adequate for ~500 nodes)
        const queue: Array<{ weight: number; name: string }> = []

        // Seed with root at maximum priority
        if (this.nodes.has('r')) {
            queue.push({ weight: Number.MAX_VALUE, name: 'r' })
        }

        const renderSet = new Set<string>()
        let numVisiblePoints = 0

        while (queue.length > 0) {
            // Pop highest weight (parents should always have higher weight than children)
            let bestIdx = 0
            for (let i = 1; i < queue.length; i++) {
                if (queue[i].weight > queue[bestIdx].weight) bestIdx = i
            }
            const entry = queue[bestIdx]
            queue.splice(bestIdx, 1)

            const node = this.nodes.get(entry.name)
            if (!node) continue

            // Frustum check
            const insideFrustum = frustum.intersectsBox(node.boundingBox)

            // Skip if not in frustum
            if (!insideFrustum) continue

            // Budget check: stop the entire traversal if this node exceeds budget.
            // Potree stops evaluating entirely instead of skipping around and causing gaps.
            if (numVisiblePoints + node.numPoints > this.pointBudget) {
                break;
            }

            // ---- This node IS visible: show it ----
            renderSet.add(node.name)
            numVisiblePoints += node.numPoints

            // If this is a proxy node, trigger hierarchy loading
            if (node.nodeType === 2 && !node.hierarchyLoaded && !node.hierarchyLoading) {
                this.loadProxyHierarchy(node)
            }

            if (!node.loaded && !node.loading && node.nodeType !== 2) {
                void this.loadNode(node)
            }

            // ---- Add children to queue if they project large enough ----
            for (let c = 0; c < 8; c++) {
                if (!(node.childMask & (1 << c))) continue
                const childName = entry.name + c
                const childNode = this.nodes.get(childName)
                if (!childNode) continue

                // Calculate screen pixel radius of child
                const center = new THREE.Vector3()
                childNode.boundingBox.getCenter(center)

                const dx = cameraPos.x - center.x
                const dy = cameraPos.y - center.y
                const dz = cameraPos.z - center.z
                const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)

                const childSize = childNode.boundingBox.getSize(new THREE.Vector3())
                const radius = childSize.length() * 0.5

                const projFactor = (0.5 * domHeight) / (slope * Math.max(distance, 0.001))
                let screenPixelRadius = radius * projFactor

                // If camera is inside the child's bounding sphere, max priority
                if (distance - radius < 0) {
                    screenPixelRadius = Number.MAX_VALUE
                }

                // Only add to queue if projected large enough
                if (screenPixelRadius < minimumNodePixelSize) continue

                queue.push({ weight: screenPixelRadius, name: childName })
            }
        }

        // Unload nodes no longer in render set
        for (const [name, node] of this.nodes) {
            if (node.loaded && !renderSet.has(name)) {
                this.unloadNode(node)
            }
        }

        this.visiblePoints = numVisiblePoints
    }

    /** Small cloud path: load ALL nodes, no frustum culling, no unloading.
     *  Behaves exactly like a regular PLY viewer — all points always visible.
     */
    private _updateVisibilityLoadAll(): void {
        let numVisible = 0
        for (const [, node] of this.nodes) {
            // Expand any proxy nodes
            if (node.nodeType === 2 && !node.hierarchyLoaded && !node.hierarchyLoading) {
                this.loadProxyHierarchy(node)
            }
            // Load every real node
            if (!node.loaded && !node.loading && node.nodeType !== 2) {
                void this.loadNode(node)
            }
            if (node.loaded) {
                numVisible += node.numPoints
            }
        }
        this.visiblePoints = numVisible
    }

    /** Load a single octree node's point data via HTTP Range request (LOD streaming) */
    private async loadNode(node: OctreeNode): Promise<void> {
        if (!this.metadata || node.loaded || node.loading) return

        // Cache hit: geometry still in memory from a previous visit → just
        // re-attach to the scene. No network, no re-parse (instant).
        if (node.points) {
            this.octreeGroup.add(node.points)
            node.loaded = true
            node.lastUsed = ++this._tick
            this.totalLoadedPoints += node.numPoints
            return
        }

        node.loading = true

        const meta = this.metadata
        const byteOffset = Number(node.byteOffset)
        const byteSize = Number(node.byteSize)

        if (byteSize <= 0) { node.loading = false; return }

        // ── Fetch ONLY this node's byte range from octree.bin ──
        // Starlette's FileResponse honours Range → returns 206 with just the
        // slice. We never hold the whole multi-GB file in memory.
        let buffer: ArrayBuffer
        try {
            const resp = await fetch(this.baseUrl + 'octree.bin', {
                headers: { 'Range': `bytes=${byteOffset}-${byteOffset + byteSize - 1}` },
                signal: this._abort.signal,
            })
            if (resp.status !== 206 && resp.status !== 200) throw new Error(`HTTP ${resp.status}`)
            buffer = await resp.arrayBuffer()
        } catch (e) {
            // Aborted on dispose (session unload) — not an error, just stop.
            if ((e as Error)?.name === 'AbortError') { node.loading = false; return }
            console.error(`[PotreeLoader] Node ${node.name}: range fetch failed`, e)
            node.loading = false
            return
        }

        // Disposed/replaced while awaiting? bail out.
        if (!this.metadata) { node.loading = false; return }

        // 206 → buffer is exactly the node slice (base 0). If a server ignored
        // Range and returned 200 (whole file), fall back to slicing at byteOffset.
        let dvBase = 0
        if (buffer.byteLength !== byteSize && buffer.byteLength >= byteOffset + byteSize) {
            dvBase = byteOffset
        }
        const dvLen = Math.min(byteSize, buffer.byteLength - dvBase)

        const bytesPerPoint = meta.attributes.reduce((sum, attr) => sum + attr.size, 0)
        if (bytesPerPoint === 0) { node.loading = false; return }

        const maxPointsFromData = Math.floor(dvLen / bytesPerPoint)
        const numPoints = Math.min(node.numPoints, maxPointsFromData)
        if (numPoints === 0) { node.loading = false; return }

        const nodeData = new DataView(buffer, dvBase, dvLen)

        let posOffset = -1
        let rgbOffset = -1
        let intensityOffset = -1
        let classOffset = -1
        let confidenceOffset = -1
        let attrOffset = 0
        for (const attr of meta.attributes) {
            if (attr.name === 'position') posOffset = attrOffset
            else if (attr.name === 'rgb') rgbOffset = attrOffset
            else if (attr.name === 'intensity') intensityOffset = attrOffset
            else if (attr.name === 'classification') classOffset = attrOffset
            else if (attr.name === 'confidence') confidenceOffset = attrOffset
            attrOffset += attr.size
        }

        if (posOffset < 0) {
            console.warn(`[PotreeLoader] No position attribute found`)
            node.loading = false
            return
        }

        // Confidence: prefer the raw `confidence` extra dim (true DA3 range, e.g.
        // 0.34..13.87) normalized to [0,1] via metadata min/max so the 0..1 slider
        // spans the real distribution. Fall back to intensity (clamped) if absent.
        const confAttr = meta.attributes.find(a => a.name === 'confidence')
        const confMin = confAttr?.min?.[0] ?? 0
        const confMax = confAttr?.max?.[0] ?? 1
        const confRange = Math.max(confMax - confMin, 1e-6)
        const hasConf = confidenceOffset >= 0 || intensityOffset >= 0

        // Extract positions and colors
        const positions = new Float32Array(numPoints * 3)
        const colors = new Float32Array(numPoints * 3)
        const classIds = new Float32Array(numPoints)
        const confidences = hasConf ? new Float32Array(numPoints) : null

        const scale = meta.scale
        const offset = meta.offset

        let validPoints = 0
        for (let i = 0; i < numPoints; i++) {
            const base = i * bytesPerPoint

            // Safety: check we won't read past the DataView
            if (base + posOffset + 12 > dvLen) break
            if (rgbOffset >= 0 && base + rgbOffset + 6 > dvLen) break

            // Potree 2.0: int32 quantized positions → int * scale + offset
            const ix = nodeData.getInt32(base + posOffset, true)
            const iy = nodeData.getInt32(base + posOffset + 4, true)
            const iz = nodeData.getInt32(base + posOffset + 8, true)

            positions[validPoints * 3] = ix * scale[0] + offset[0]
            positions[validPoints * 3 + 1] = iy * scale[1] + offset[1]
            positions[validPoints * 3 + 2] = iz * scale[2] + offset[2]

            // RGB: uint16 values normalized to [0, 1]
            if (rgbOffset >= 0) {
                const r = nodeData.getUint16(base + rgbOffset, true)
                const g = nodeData.getUint16(base + rgbOffset + 2, true)
                const b = nodeData.getUint16(base + rgbOffset + 4, true)
                colors[validPoints * 3] = r / 65535
                colors[validPoints * 3 + 1] = g / 65535
                colors[validPoints * 3 + 2] = b / 65535
            } else {
                colors[validPoints * 3] = 0.7
                colors[validPoints * 3 + 1] = 0.7
                colors[validPoints * 3 + 2] = 0.7
            }

            // Classification (segment ID: 0=unsegmented, 1..N=segment, -1=always visible)
            if (this._forceClassId !== undefined) {
                classIds[validPoints] = this._forceClassId
            } else if (classOffset >= 0 && base + classOffset + 1 <= dvLen) {
                classIds[validPoints] = nodeData.getUint8(base + classOffset)
            } else {
                classIds[validPoints] = -1  // no classification → always visible (e.g. sábana)
            }

            // Confidence → normalized [0,1] for the adjustable slider
            if (confidences) {
                if (confidenceOffset >= 0 && base + confidenceOffset + 4 <= dvLen) {
                    const c = nodeData.getFloat32(base + confidenceOffset, true)
                    confidences[validPoints] = (c - confMin) / confRange
                } else if (intensityOffset >= 0 && base + intensityOffset + 2 <= dvLen) {
                    confidences[validPoints] = nodeData.getUint16(base + intensityOffset, true) / 65535
                }
            }

            validPoints++
        }

        if (validPoints === 0) {
            node.loading = false
            return
        }

        // Create Three.js geometry (trim arrays to actual valid count)
        const geometry = new THREE.BufferGeometry()
        geometry.setAttribute('position', new THREE.BufferAttribute(positions.subarray(0, validPoints * 3), 3))
        geometry.setAttribute('color', new THREE.BufferAttribute(colors.subarray(0, validPoints * 3), 3))
        geometry.setAttribute('classId', new THREE.BufferAttribute(classIds.subarray(0, validPoints), 1))
        if (confidences) {
            geometry.setAttribute('confidence', new THREE.BufferAttribute(confidences.subarray(0, validPoints), 1))
            this._hasConfidence = true
        }
        geometry.computeBoundingSphere()

        // Use the shared material from the Viewport
        const points = new THREE.Points(geometry, this.material)
        points.name = `potree-node-${node.name}`

        // For small clouds (< pointBudget): disable Three.js frustum culling.
        // Three.js culls by bounding sphere which is unreliable after transforms
        // (floor rotation). Since we load all nodes anyway, let the GPU handle it.
        if (this.metadata && this.metadata.points <= this.pointBudget) {
            points.frustumCulled = false
        }

        this.octreeGroup.add(points)
        node.points = points
        node.loaded = true
        node.loading = false
        node.lastUsed = ++this._tick
        this.totalLoadedPoints += node.numPoints
        this._cachedPoints += node.numPoints
        this._evictCache()
    }

    /** Remove a node from the scene but KEEP its geometry cached (no dispose),
     *  so re-entering its region re-shows it instantly without an HTTP re-fetch. */
    private unloadNode(node: OctreeNode): void {
        if (!node.loaded || !node.points) return

        this.octreeGroup.remove(node.points)
        node.loaded = false
        this.totalLoadedPoints -= node.numPoints
        // geometry stays in node.points (cached); _evictCache() frees it later
    }

    /** LRU eviction: when cached (off-screen) geometry exceeds the cap, dispose
     *  the least-recently-shown NOT-in-scene nodes until back under budget. */
    private _evictCache(): void {
        const cap = Math.max(this.pointBudget * 2, 15_000_000)
        if (this._cachedPoints <= cap) return

        const evictable: OctreeNode[] = []
        for (const [, n] of this.nodes) {
            if (n.points && !n.loaded) evictable.push(n)
        }
        evictable.sort((a, b) => a.lastUsed - b.lastUsed)  // oldest first

        for (const n of evictable) {
            if (this._cachedPoints <= cap) break
            n.points!.geometry.dispose()
            n.points = null
            this._cachedPoints -= n.numPoints
        }
    }

    /** Remove everything and reset (disposes both in-scene and cached geometry) */
    dispose(): void {
        // Cancel all in-flight fetches so they don't clog the browser's
        // connection pool and starve the next session's load.
        this._abort.abort()
        for (const [, node] of this.nodes) {
            if (node.points) {
                this.octreeGroup.remove(node.points)
                node.points.geometry.dispose()
                node.points = null
            }
            node.loaded = false
        }
        this.scene.remove(this.octreeGroup)
        this.nodes.clear()
        this.metadata = null
        this.hierarchyData = null
        this.totalLoadedPoints = 0
        this._cachedPoints = 0
        this._loading = false
    }

    /** Get bounding box of the whole cloud (local space, from metadata) */
    getBoundingBox(): THREE.Box3 | null {
        if (!this.metadata) return null
        return new THREE.Box3(
            new THREE.Vector3(...this.metadata.boundingBox.min),
            new THREE.Vector3(...this.metadata.boundingBox.max),
        )
    }

    /** Get world-space TIGHT bounding box.
     *  Uses the position attribute's min/max from metadata (actual point extent),
     *  NOT the octree boundingBox which is always a perfect cube. */
    getWorldBoundingBox(): THREE.Box3 | null {
        if (!this.metadata) return null

        // Find position attribute — its min/max is the tight bbox
        const posAttr = this.metadata.attributes.find(a => a.name === 'position')
        let localBox: THREE.Box3
        if (posAttr?.min && posAttr?.max) {
            localBox = new THREE.Box3(
                new THREE.Vector3(posAttr.min[0], posAttr.min[1], posAttr.min[2]),
                new THREE.Vector3(posAttr.max[0], posAttr.max[1], posAttr.max[2]),
            )
        } else {
            // Fallback to octree cube if position attr has no min/max
            localBox = this.getBoundingBox()!
        }

        // Transform through the octreeGroup's world matrix (floor transform)
        const m = this.octreeGroup.matrixWorld
        const corners = [
            new THREE.Vector3(localBox.min.x, localBox.min.y, localBox.min.z),
            new THREE.Vector3(localBox.min.x, localBox.min.y, localBox.max.z),
            new THREE.Vector3(localBox.min.x, localBox.max.y, localBox.min.z),
            new THREE.Vector3(localBox.min.x, localBox.max.y, localBox.max.z),
            new THREE.Vector3(localBox.max.x, localBox.min.y, localBox.min.z),
            new THREE.Vector3(localBox.max.x, localBox.min.y, localBox.max.z),
            new THREE.Vector3(localBox.max.x, localBox.max.y, localBox.min.z),
            new THREE.Vector3(localBox.max.x, localBox.max.y, localBox.max.z),
        ]
        const worldBox = new THREE.Box3()
        for (const c of corners) {
            c.applyMatrix4(m)
            worldBox.expandByPoint(c)
        }
        return worldBox
    }

    /** Get visible point count (after LOD culling) */
    getVisiblePointCount(): number {
        return this.visiblePoints
    }

    /** Whether the loaded cloud has per-point confidence data */
    get hasConfidence(): boolean {
        return this._hasConfidence
    }
}
