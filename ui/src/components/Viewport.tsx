/**
 * STAC Build — 3D Viewport Component
 * Three.js-based point cloud renderer
 * Hernán Barreto — Ingerop IN3
 */
import { useRef, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { PotreeOctreeLoader } from './PotreeLoader'

type Tool = 'navigate' | 'measure-distance' | 'measure-angle' | 'section-box'

interface ViewportProps {
    pointSize: number
    activeSession: string | null
    activeTool: Tool
    onPointCount: (count: number) => void
    onFps: (fps: number) => void
    onStatusMessage?: (msg: string) => void
    onSegments?: (segments: SegmentInstance[]) => void
    onPipelineProgress?: (data: Record<string, unknown>) => void
}

export interface SegmentInstance {
    key: string
    label: string
    color: string
    totalPoints: number
    visible: boolean
}

export interface ViewportHandle {
    sendCommand: (cmd: Record<string, unknown>) => void
    toggleOBB: (key: string, visible: boolean) => void
    clearMeasurements: () => void
    resetSectionBox: () => void
    resetCamera: () => void
}

// Vertex shader — matches FusionRenderer.js point size formula
const vertexShader = `
  attribute float classId;
  varying float vClassId;
  varying vec3 vColor;
  varying vec3 vWorldPos;
  uniform float pointSize;

  void main() {
    vClassId = classId;
    vColor = color;
    vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
    
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    
    // Same formula as standalone viewer: pointScale / depth
    float depth = -mvPosition.z;
    float size = pointSize / depth;
    gl_PointSize = clamp(size, 1.0, 5.0);
  }
`

// Fragment shader — circular points with EDL-like shading + section box clipping
const fragmentShader = `
  varying float vClassId;
  varying vec3 vColor;
  varying vec3 vWorldPos;
  uniform float highlightIntensity;
  uniform bool sectionBoxEnabled;
  uniform vec3 sectionBoxMin;
  uniform vec3 sectionBoxMax;

  void main() {
    // Section box clipping
    if (sectionBoxEnabled) {
      if (vWorldPos.x < sectionBoxMin.x || vWorldPos.x > sectionBoxMax.x ||
          vWorldPos.y < sectionBoxMin.y || vWorldPos.y > sectionBoxMax.y ||
          vWorldPos.z < sectionBoxMin.z || vWorldPos.z > sectionBoxMax.z) {
        discard;
      }
    }
    
    vec2 centered = gl_PointCoord - 0.5;
    float dist = length(centered);
    if (dist > 0.5) discard;
    
    float alpha = 1.0 - smoothstep(0.35, 0.5, dist);
    
    vec3 finalColor = vColor * 0.85;
    
    gl_FragColor = vec4(finalColor, alpha);
  }
`

// ── Measurement helpers (outside component) ──
function createTextSprite(text: string, color: string = '#ffffff'): THREE.Sprite {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')!
    canvas.width = 256
    canvas.height = 64
    ctx.fillStyle = 'rgba(0,0,0,0.7)'
    ctx.roundRect(0, 0, 256, 64, 8)
    ctx.fill()
    ctx.font = 'bold 28px Inter, Arial, sans-serif'
    ctx.fillStyle = color
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(text, 128, 32)

    const texture = new THREE.CanvasTexture(canvas)
    texture.needsUpdate = true
    const mat = new THREE.SpriteMaterial({ map: texture, depthTest: false, sizeAttenuation: true })
    const sprite = new THREE.Sprite(mat)
    sprite.scale.set(0.4, 0.1, 1)
    return sprite
}

function createMarker(position: THREE.Vector3, color: number = 0xff4444): THREE.Mesh {
    const geom = new THREE.SphereGeometry(0.008, 12, 12)
    const mat = new THREE.MeshBasicMaterial({ color, depthTest: false })
    const mesh = new THREE.Mesh(geom, mat)
    mesh.position.copy(position)
    mesh.renderOrder = 999
    return mesh
}

function createMeasurementLine(points: THREE.Vector3[], color: number = 0x00ff88): THREE.Line {
    const geom = new THREE.BufferGeometry().setFromPoints(points)
    const mat = new THREE.LineBasicMaterial({ color, depthTest: false, linewidth: 2 })
    const line = new THREE.Line(geom, mat)
    line.renderOrder = 999
    return line
}

function createArc(
    center: THREE.Vector3, v1: THREE.Vector3, v2: THREE.Vector3,
    angleDeg: number, radius: number = 0.1
): THREE.Line {
    const n1 = v1.clone().normalize()
    const n2 = v2.clone().normalize()
    const segments = 24
    const angleRad = angleDeg * Math.PI / 180
    const pts: THREE.Vector3[] = []
    for (let i = 0; i <= segments; i++) {
        const t = i / segments
        // Slerp-like interpolation in the plane of the two vectors
        const a = Math.sin((1 - t) * angleRad) / Math.sin(angleRad)
        const b = Math.sin(t * angleRad) / Math.sin(angleRad)
        if (!isFinite(a) || !isFinite(b)) continue
        const pt = new THREE.Vector3()
            .addScaledVector(n1, a * radius)
            .addScaledVector(n2, b * radius)
            .add(center)
        pts.push(pt)
    }
    if (pts.length < 2) pts.push(center.clone(), center.clone())
    return createMeasurementLine(pts, 0xffaa00)
}

// ── Measurement data types ──
interface Measurement {
    type: 'distance' | 'angle'
    objects: THREE.Object3D[]  // markers, lines, labels, arcs
}

const Viewport = forwardRef<ViewportHandle, ViewportProps>(function Viewport(
    { pointSize, activeSession, activeTool, onPointCount, onFps, onStatusMessage, onSegments, onPipelineProgress },
    ref
) {
    const containerRef = useRef<HTMLDivElement>(null)
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
    const sceneRef = useRef<THREE.Scene | null>(null)
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
    const controlsRef = useRef<OrbitControls | null>(null)
    const pointCloudRef = useRef<THREE.Points | null>(null)
    const materialRef = useRef<THREE.ShaderMaterial | null>(null)
    const animFrameRef = useRef<number>(0)
    const wsRef = useRef<WebSocket | null>(null)
    const totalPointsRef = useRef(0)
    const geometryRef = useRef<THREE.BufferGeometry | null>(null)
    const obbGroupRef = useRef<THREE.Group | null>(null)
    const obbMapRef = useRef<Map<string, THREE.Object3D>>(new Map())

    // Measurement state
    const measureGroupRef = useRef<THREE.Group | null>(null)
    const pendingPointsRef = useRef<THREE.Vector3[]>([])
    const pendingMarkersRef = useRef<THREE.Mesh[]>([])
    const pendingLineRef = useRef<THREE.Line | null>(null)
    const measurementsRef = useRef<Measurement[]>([])
    const raycasterRef = useRef(new THREE.Raycaster())
    const activeToolRef = useRef(activeTool)
    const hoverHighlightRef = useRef<THREE.Group | null>(null)
    const potreeLoaderRef = useRef<PotreeOctreeLoader | null>(null)
    const lastLodUpdateRef = useRef(0)
    const floorTransformRef = useRef<THREE.Matrix4 | null>(null)

    // Keep activeToolRef in sync with prop
    useEffect(() => { activeToolRef.current = activeTool }, [activeTool])

    // Toggle OrbitControls left-button based on active tool
    useEffect(() => {
        const controls = controlsRef.current
        if (!controls) return
        if (activeTool === 'measure-distance' || activeTool === 'measure-angle' || activeTool === 'section-box') {
            controls.mouseButtons.LEFT = undefined as unknown as THREE.MOUSE
        } else {
            controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE
        }
    }, [activeTool])

    // Clear all measurements
    const clearAllMeasurements = useCallback(() => {
        const group = measureGroupRef.current
        if (!group) return
        while (group.children.length > 0) {
            const child = group.children[0]
            group.remove(child)
            if ((child as THREE.Mesh).geometry) (child as THREE.Mesh).geometry.dispose()
            if ((child as THREE.Mesh).material) {
                const mat = (child as THREE.Mesh).material
                if (Array.isArray(mat)) mat.forEach(m => m.dispose())
                else (mat as THREE.Material).dispose()
            }
        }
        pendingPointsRef.current = []
        pendingMarkersRef.current = []
        pendingLineRef.current = null
        measurementsRef.current = []
    }, [])

    // Cancel in-progress measurement
    const cancelPending = useCallback(() => {
        const group = measureGroupRef.current
        if (!group) return
        // Remove pending markers
        for (const m of pendingMarkersRef.current) {
            group.remove(m)
            m.geometry.dispose()
                ; (m.material as THREE.Material).dispose()
        }
        // Remove pending preview line
        if (pendingLineRef.current) {
            group.remove(pendingLineRef.current)
            pendingLineRef.current.geometry.dispose()
                ; (pendingLineRef.current.material as THREE.Material).dispose()
            pendingLineRef.current = null
        }
        pendingPointsRef.current = []
        pendingMarkersRef.current = []
    }, [])

    // Finalize a distance measurement
    const finalizeDistance = useCallback((p1: THREE.Vector3, p2: THREE.Vector3) => {
        const group = measureGroupRef.current
        if (!group) return
        const dist = p1.distanceTo(p2)
        const label = dist < 1 ? `${(dist * 100).toFixed(1)} cm` : `${dist.toFixed(3)} m`

        const line = createMeasurementLine([p1, p2], 0x00ff88)
        const mid = p1.clone().add(p2).multiplyScalar(0.5)
        const sprite = createTextSprite(label, '#00ff88')
        sprite.position.copy(mid)
        sprite.position.y += 0.05

        group.add(line)
        group.add(sprite)

        const measurement: Measurement = {
            type: 'distance',
            objects: [...pendingMarkersRef.current, line, sprite]
        }
        measurementsRef.current.push(measurement)

        // Reset pending
        pendingPointsRef.current = []
        pendingMarkersRef.current = []
        pendingLineRef.current = null

        if (onStatusMessage) onStatusMessage(`Distance: ${label}`)
    }, [onStatusMessage])

    // Finalize an angle measurement
    const finalizeAngle = useCallback((p1: THREE.Vector3, vertex: THREE.Vector3, p3: THREE.Vector3) => {
        const group = measureGroupRef.current
        if (!group) return

        const v1 = p1.clone().sub(vertex)
        const v2 = p3.clone().sub(vertex)
        const angleRad = v1.angleTo(v2)
        const angleDeg = THREE.MathUtils.radToDeg(angleRad)

        const line1 = createMeasurementLine([vertex, p1], 0xffaa00)
        const line2 = createMeasurementLine([vertex, p3], 0xffaa00)
        const arc = createArc(vertex, v1, v2, angleDeg, 0.15)

        const label = `${angleDeg.toFixed(1)}\u00B0`
        const sprite = createTextSprite(label, '#ffaa00')
        const labelDir = v1.clone().normalize().add(v2.clone().normalize()).normalize()
        sprite.position.copy(vertex).addScaledVector(labelDir, 0.2)

        group.add(line1)
        group.add(line2)
        group.add(arc)
        group.add(sprite)

        const measurement: Measurement = {
            type: 'angle',
            objects: [...pendingMarkersRef.current, line1, line2, arc, sprite]
        }
        measurementsRef.current.push(measurement)

        pendingPointsRef.current = []
        pendingMarkersRef.current = []
        pendingLineRef.current = null

        if (onStatusMessage) onStatusMessage(`Angle: ${label}`)
    }, [onStatusMessage])

    // Hover highlight for measurement — shows which point would be picked
    const handleMeasureHover = useCallback((event: MouseEvent) => {
        const tool = activeToolRef.current
        if (tool !== 'measure-distance' && tool !== 'measure-angle') {
            if (hoverHighlightRef.current) hoverHighlightRef.current.visible = false
            return
        }

        const renderer = rendererRef.current
        const camera = cameraRef.current
        const pointCloud = pointCloudRef.current
        const scene = sceneRef.current
        if (!renderer || !camera || !pointCloud || !scene) return

        const rect = renderer.domElement.getBoundingClientRect()
        const mouse = new THREE.Vector2(
            ((event.clientX - rect.left) / rect.width) * 2 - 1,
            -((event.clientY - rect.top) / rect.height) * 2 + 1
        )

        const raycaster = raycasterRef.current
        raycaster.params.Points = { threshold: 0.01 }
        raycaster.setFromCamera(mouse, camera)

        // Raycast against both legacy pointCloud and Potree octree nodes
        const targets: THREE.Object3D[] = [pointCloud]
        if (potreeLoaderRef.current) {
            const octreeGroup = sceneRef.current?.getObjectByName('potree-octree')
            if (octreeGroup) targets.push(...octreeGroup.children)
        }
        const intersects = raycaster.intersectObjects(targets)
        // Filter out section-box-clipped points
        const hit = intersects.find(i => {
            if (!sectionBoxActiveRef.current) return true
            const p = i.point
            const bMin = sectionBoxMinRef.current
            const bMax = sectionBoxMaxRef.current
            return p.x >= bMin.x && p.x <= bMax.x &&
                p.y >= bMin.y && p.y <= bMax.y &&
                p.z >= bMin.z && p.z <= bMax.z
        })
        if (!hit) {
            if (hoverHighlightRef.current) hoverHighlightRef.current.visible = false
            return
        }

        const hitPoint = hit.point

        // Create hover highlight lazily
        if (!hoverHighlightRef.current) {
            const group = new THREE.Group()
            // Outer ring
            const ringGeom = new THREE.RingGeometry(0.010, 0.016, 24)
            const ringMat = new THREE.MeshBasicMaterial({
                color: 0x00ffff, side: THREE.DoubleSide,
                transparent: true, opacity: 0.7, depthTest: false
            })
            const ring = new THREE.Mesh(ringGeom, ringMat)
            ring.renderOrder = 1000
            group.add(ring)
            // Center dot
            const dotGeom = new THREE.SphereGeometry(0.004, 8, 8)
            const dotMat = new THREE.MeshBasicMaterial({
                color: 0xffffff, depthTest: false,
                transparent: true, opacity: 0.9
            })
            const dot = new THREE.Mesh(dotGeom, dotMat)
            dot.renderOrder = 1001
            group.add(dot)
            group.name = 'hoverHighlight'
            scene.add(group)
            hoverHighlightRef.current = group
        }

        const hl = hoverHighlightRef.current
        hl.position.copy(hitPoint)
        hl.visible = true
        // Billboard — always face camera
        hl.quaternion.copy(camera.quaternion)
    }, [])

    // Handle click for measurement
    const handleMeasureClick = useCallback((event: MouseEvent) => {
        const tool = activeToolRef.current
        if (tool !== 'measure-distance' && tool !== 'measure-angle') return
        if (event.button !== 0) return  // Left click only

        const renderer = rendererRef.current
        const camera = cameraRef.current
        const pointCloud = pointCloudRef.current
        const group = measureGroupRef.current
        if (!renderer || !camera || !pointCloud || !group) return

        const rect = renderer.domElement.getBoundingClientRect()
        const mouse = new THREE.Vector2(
            ((event.clientX - rect.left) / rect.width) * 2 - 1,
            -((event.clientY - rect.top) / rect.height) * 2 + 1
        )

        const raycaster = raycasterRef.current
        raycaster.params.Points = { threshold: 0.01 }
        raycaster.setFromCamera(mouse, camera)

        // Raycast against both legacy pointCloud and Potree octree nodes
        const targets: THREE.Object3D[] = [pointCloud]
        if (potreeLoaderRef.current) {
            const octreeGroup = sceneRef.current?.getObjectByName('potree-octree')
            if (octreeGroup) targets.push(...octreeGroup.children)
        }
        const intersects = raycaster.intersectObjects(targets)
        // Filter out section-box-clipped points
        const hit = intersects.find(i => {
            if (!sectionBoxActiveRef.current) return true
            const p = i.point
            const bMin = sectionBoxMinRef.current
            const bMax = sectionBoxMaxRef.current
            return p.x >= bMin.x && p.x <= bMax.x &&
                p.y >= bMin.y && p.y <= bMax.y &&
                p.z >= bMin.z && p.z <= bMax.z
        })
        if (!hit) return

        const hitPoint = hit.point.clone()
        pendingPointsRef.current.push(hitPoint)

        // Place marker
        const markerColor = tool === 'measure-distance' ? 0x00ff88 : 0xffaa00
        const marker = createMarker(hitPoint, markerColor)
        group.add(marker)
        pendingMarkersRef.current.push(marker)

        const nPoints = pendingPointsRef.current.length

        // Update preview line
        if (nPoints >= 2) {
            if (pendingLineRef.current) {
                group.remove(pendingLineRef.current)
                pendingLineRef.current.geometry.dispose()
                    ; (pendingLineRef.current.material as THREE.Material).dispose()
            }
            const lineColor = tool === 'measure-distance' ? 0x00ff88 : 0xffaa00
            const previewLine = createMeasurementLine(
                pendingPointsRef.current.slice(),
                lineColor
            )
            group.add(previewLine)
            pendingLineRef.current = previewLine
        }

        // Finalize measurement
        if (tool === 'measure-distance' && nPoints === 2) {
            // Remove preview line (finalizeDistance creates its own)
            if (pendingLineRef.current) {
                group.remove(pendingLineRef.current)
                pendingLineRef.current.geometry.dispose()
                    ; (pendingLineRef.current.material as THREE.Material).dispose()
                pendingLineRef.current = null
            }
            finalizeDistance(pendingPointsRef.current[0], pendingPointsRef.current[1])
        } else if (tool === 'measure-angle' && nPoints === 3) {
            if (pendingLineRef.current) {
                group.remove(pendingLineRef.current)
                pendingLineRef.current.geometry.dispose()
                    ; (pendingLineRef.current.material as THREE.Material).dispose()
                pendingLineRef.current = null
            }
            finalizeAngle(
                pendingPointsRef.current[0],
                pendingPointsRef.current[1],
                pendingPointsRef.current[2]
            )
        }
    }, [finalizeDistance, finalizeAngle])

    // ── Section Box State ──
    const sectionBoxGroupRef = useRef<THREE.Group | null>(null)
    const sectionBoxWireRef = useRef<THREE.LineSegments | null>(null)
    const sectionBoxHandlesRef = useRef<THREE.Mesh[]>([])
    const sectionBoxMinRef = useRef(new THREE.Vector3(-100, -100, -100))
    const sectionBoxMaxRef = useRef(new THREE.Vector3(100, 100, 100))
    const sectionBoxActiveRef = useRef(false)
    const isDraggingHandleRef = useRef(false)
    const dragHandleIndexRef = useRef(-1)
    const dragPlaneRef = useRef(new THREE.Plane())
    const dragAxisRef = useRef(0)   // 0=x, 1=y, 2=z
    const dragSideRef = useRef(0)   // 0=min, 1=max

    // Handle axis/face definitions:  [axisIndex, side(min=0/max=1)]
    const HANDLE_DEFS: [number, number][] = [
        [0, 0], [0, 1],  // X min, X max
        [1, 0], [1, 1],  // Y min, Y max
        [2, 0], [2, 1],  // Z min, Z max
    ]
    const HANDLE_COLORS = [0xff4444, 0xff4444, 0x44ff44, 0x44ff44, 0x4488ff, 0x4488ff]

    const updateSectionBoxWireframe = useCallback(() => {
        const wire = sectionBoxWireRef.current
        if (!wire) return
        const bMin = sectionBoxMinRef.current
        const bMax = sectionBoxMaxRef.current
        const cx = (bMin.x + bMax.x) / 2
        const cy = (bMin.y + bMax.y) / 2
        const cz = (bMin.z + bMax.z) / 2
        wire.position.set(cx, cy, cz)
        wire.scale.set(bMax.x - bMin.x, bMax.y - bMin.y, bMax.z - bMin.z)
    }, [])

    const updateSectionBoxHandles = useCallback(() => {
        const handles = sectionBoxHandlesRef.current
        if (handles.length !== 6) return
        const bMin = sectionBoxMinRef.current
        const bMax = sectionBoxMaxRef.current
        const cx = (bMin.x + bMax.x) / 2
        const cy = (bMin.y + bMax.y) / 2
        const cz = (bMin.z + bMax.z) / 2
        handles[0].position.set(bMin.x, cy, cz)  // X min
        handles[1].position.set(bMax.x, cy, cz)  // X max
        handles[2].position.set(cx, bMin.y, cz)  // Y min
        handles[3].position.set(cx, bMax.y, cz)  // Y max
        handles[4].position.set(cx, cy, bMin.z)  // Z min
        handles[5].position.set(cx, cy, bMax.z)  // Z max
    }, [])

    const updateSectionBoxUniforms = useCallback(() => {
        const mat = materialRef.current
        if (!mat) return
        mat.uniforms.sectionBoxEnabled.value = sectionBoxActiveRef.current
        mat.uniforms.sectionBoxMin.value.copy(sectionBoxMinRef.current)
        mat.uniforms.sectionBoxMax.value.copy(sectionBoxMaxRef.current)
    }, [])

    const createSectionBox = useCallback(() => {
        const scene = sceneRef.current
        if (!scene) return

        // Remove old section box if exists
        if (sectionBoxGroupRef.current) {
            scene.remove(sectionBoxGroupRef.current)
        }

        const group = new THREE.Group()
        group.name = 'sectionBox'

        // Compute bounds from point cloud (Potree or legacy)
        const pad = 0.1
        const potreeBBox = potreeLoaderRef.current?.getBoundingBox()
        if (potreeBBox) {
            sectionBoxMinRef.current.copy(potreeBBox.min).addScalar(-pad)
            sectionBoxMaxRef.current.copy(potreeBBox.max).addScalar(pad)
        } else {
            const geom = geometryRef.current
            if (geom) {
                geom.computeBoundingBox()
                if (geom.boundingBox) {
                    sectionBoxMinRef.current.copy(geom.boundingBox.min).addScalar(-pad)
                    sectionBoxMaxRef.current.copy(geom.boundingBox.max).addScalar(pad)
                }
            }
        }

        // Wireframe box (unit cube scaled/positioned)
        const boxGeom = new THREE.BoxGeometry(1, 1, 1)
        const edges = new THREE.EdgesGeometry(boxGeom)
        const wireMat = new THREE.LineBasicMaterial({ color: 0x00ddff, linewidth: 1, transparent: true, opacity: 0.6 })
        const wireframe = new THREE.LineSegments(edges, wireMat)
        wireframe.renderOrder = 998
        group.add(wireframe)
        sectionBoxWireRef.current = wireframe
        boxGeom.dispose()

        // 6 face handles (small spheres)
        const handles: THREE.Mesh[] = []
        for (let i = 0; i < 6; i++) {
            const hGeom = new THREE.SphereGeometry(0.04, 12, 12)
            const hMat = new THREE.MeshBasicMaterial({ color: HANDLE_COLORS[i], depthTest: false })
            const handle = new THREE.Mesh(hGeom, hMat)
            handle.renderOrder = 999
            handle.userData.handleIndex = i
            group.add(handle)
            handles.push(handle)
        }
        sectionBoxHandlesRef.current = handles

        scene.add(group)
        sectionBoxGroupRef.current = group

        sectionBoxActiveRef.current = true
        updateSectionBoxWireframe()
        updateSectionBoxHandles()
        updateSectionBoxUniforms()
    }, [updateSectionBoxWireframe, updateSectionBoxHandles, updateSectionBoxUniforms])

    const destroySectionBox = useCallback(() => {
        const scene = sceneRef.current
        const group = sectionBoxGroupRef.current
        if (scene && group) {
            scene.remove(group)
            // Dispose children
            group.traverse((child) => {
                if ((child as THREE.Mesh).geometry) (child as THREE.Mesh).geometry.dispose()
                if ((child as THREE.Mesh).material) {
                    const mat = (child as THREE.Mesh).material
                    if (Array.isArray(mat)) mat.forEach(m => m.dispose())
                    else (mat as THREE.Material).dispose()
                }
            })
        }
        sectionBoxGroupRef.current = null
        sectionBoxWireRef.current = null
        sectionBoxHandlesRef.current = []
        sectionBoxActiveRef.current = false
        updateSectionBoxUniforms()
    }, [updateSectionBoxUniforms])

    // Section box handle drag handlers
    const handleSectionMouseDown = useCallback((event: MouseEvent) => {
        if (activeToolRef.current !== 'section-box') return
        if (event.button !== 0) return
        const handles = sectionBoxHandlesRef.current
        if (handles.length === 0) return

        const renderer = rendererRef.current
        const camera = cameraRef.current
        if (!renderer || !camera) return

        const rect = renderer.domElement.getBoundingClientRect()
        const mouse = new THREE.Vector2(
            ((event.clientX - rect.left) / rect.width) * 2 - 1,
            -((event.clientY - rect.top) / rect.height) * 2 + 1
        )

        const raycaster = raycasterRef.current
        raycaster.params.Points = { threshold: 0 }  // Don't raycast against points here
        raycaster.setFromCamera(mouse, camera)

        const intersects = raycaster.intersectObjects(handles)
        if (intersects.length === 0) return

        event.stopPropagation()
        event.preventDefault()

        const hitHandle = intersects[0].object
        const handleIdx = hitHandle.userData.handleIndex as number
        dragHandleIndexRef.current = handleIdx

        const [axis] = HANDLE_DEFS[handleIdx]
        dragAxisRef.current = axis
        dragSideRef.current = HANDLE_DEFS[handleIdx][1]

        // Create a drag plane perpendicular to screen through handle position
        const handlePos = hitHandle.position.clone()
        const cameraDir = camera.getWorldDirection(new THREE.Vector3())
        // Use a plane that contains the handle and faces the camera, but project on axis
        const planeNormal = cameraDir.clone()
        dragPlaneRef.current.setFromNormalAndCoplanarPoint(planeNormal, handlePos)

        isDraggingHandleRef.current = true
        // Disable orbit controls during drag
        if (controlsRef.current) controlsRef.current.enabled = false
    }, [])

    const handleSectionMouseMove = useCallback((event: MouseEvent) => {
        if (!isDraggingHandleRef.current) return

        const renderer = rendererRef.current
        const camera = cameraRef.current
        if (!renderer || !camera) return

        const rect = renderer.domElement.getBoundingClientRect()
        const mouse = new THREE.Vector2(
            ((event.clientX - rect.left) / rect.width) * 2 - 1,
            -((event.clientY - rect.top) / rect.height) * 2 + 1
        )

        const raycaster = raycasterRef.current
        raycaster.setFromCamera(mouse, camera)

        const intersectPoint = new THREE.Vector3()
        if (!raycaster.ray.intersectPlane(dragPlaneRef.current, intersectPoint)) return

        const axis = dragAxisRef.current
        const side = dragSideRef.current
        const axisVal = axis === 0 ? intersectPoint.x : axis === 1 ? intersectPoint.y : intersectPoint.z

        // Clamp: don't let min pass max and vice versa
        const minVec = sectionBoxMinRef.current
        const maxVec = sectionBoxMaxRef.current
        const minArr = [minVec.x, minVec.y, minVec.z]
        const maxArr = [maxVec.x, maxVec.y, maxVec.z]

        if (side === 0) {
            // Moving min face
            minArr[axis] = Math.min(axisVal, maxArr[axis] - 0.05)
        } else {
            // Moving max face
            maxArr[axis] = Math.max(axisVal, minArr[axis] + 0.05)
        }

        sectionBoxMinRef.current.set(minArr[0], minArr[1], minArr[2])
        sectionBoxMaxRef.current.set(maxArr[0], maxArr[1], maxArr[2])

        updateSectionBoxWireframe()
        updateSectionBoxHandles()
        updateSectionBoxUniforms()
    }, [updateSectionBoxWireframe, updateSectionBoxHandles, updateSectionBoxUniforms])

    const handleSectionMouseUp = useCallback(() => {
        if (!isDraggingHandleRef.current) return
        isDraggingHandleRef.current = false
        dragHandleIndexRef.current = -1
        // Re-enable orbit controls
        if (controlsRef.current) controlsRef.current.enabled = true
    }, [])

    // Toggle section box visibility when tool changes
    useEffect(() => {
        if (activeTool === 'section-box') {
            if (!sectionBoxGroupRef.current) {
                createSectionBox()
            } else {
                sectionBoxGroupRef.current.visible = true
            }
        } else {
            if (sectionBoxGroupRef.current) {
                // Hide handles but keep clipping active
                sectionBoxGroupRef.current.visible = false
            }
        }
    }, [activeTool, createSectionBox])

    // Expose sendCommand + toggleOBB + clearMeasurements + resetSectionBox to parent via ref
    useImperativeHandle(ref, () => ({
        sendCommand: (cmd: Record<string, unknown>) => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify(cmd))
            }
        },
        toggleOBB: (key: string, visible: boolean) => {
            const mesh = obbMapRef.current.get(key)
            if (mesh) mesh.visible = visible
        },
        clearMeasurements: clearAllMeasurements,
        resetSectionBox: destroySectionBox,
        resetCamera: () => {
            const cam = cameraRef.current
            const ctrl = controlsRef.current
            if (cam) {
                cam.position.set(5, 5, 5)
                cam.lookAt(0, 0, 0)
            }
            if (ctrl) {
                ctrl.target.set(0, 0, 0)
                ctrl.update()
            }
        }
    }))

    // FPS counter
    const fpsFramesRef = useRef(0)
    const fpsTimeRef = useRef(performance.now())

    // Initialize Three.js scene
    useEffect(() => {
        const container = containerRef.current
        if (!container) return

        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
        renderer.setPixelRatio(window.devicePixelRatio)
        renderer.setSize(container.clientWidth, container.clientHeight)
        renderer.setClearColor(0x0d1117, 1)
        container.appendChild(renderer.domElement)
        rendererRef.current = renderer

        // Scene
        const scene = new THREE.Scene()
        scene.fog = new THREE.FogExp2(0x0d1117, 0.002)
        sceneRef.current = scene

        // OBB group for segmentation bounding boxes
        const obbGroup = new THREE.Group()
        obbGroup.name = 'obbGroup'
        scene.add(obbGroup)
        obbGroupRef.current = obbGroup

        // Measurement overlay group
        const measureGroup = new THREE.Group()
        measureGroup.name = 'measureGroup'
        scene.add(measureGroup)
        measureGroupRef.current = measureGroup

        // Grid helper
        const gridHelper = new THREE.GridHelper(20, 40, 0x252d3a, 0x1c2333)
        scene.add(gridHelper)

        // Axes helper
        const axesHelper = new THREE.AxesHelper(2)
        scene.add(axesHelper)

        // Camera
        const camera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.01,
            1000
        )
        camera.position.set(5, 5, 5)
        camera.lookAt(0, 0, 0)
        cameraRef.current = camera

        // Controls
        const controls = new OrbitControls(camera, renderer.domElement)
        controls.enableDamping = true
        controls.dampingFactor = 0.08
        controls.screenSpacePanning = true
        controls.minDistance = 0.00001
        controls.maxDistance = 500
        controls.maxPolarAngle = Math.PI
        controls.zoomSpeed = 3.0
        controlsRef.current = controls

        // Shader material for point cloud
        const material = new THREE.ShaderMaterial({
            vertexShader,
            fragmentShader,
            uniforms: {
                pointSize: { value: pointSize },
                highlightIntensity: { value: 0.5 },
                time: { value: 0 },
                sectionBoxEnabled: { value: false },
                sectionBoxMin: { value: new THREE.Vector3(-100, -100, -100) },
                sectionBoxMax: { value: new THREE.Vector3(100, 100, 100) },
            },
            vertexColors: true,
            transparent: true,
            depthWrite: true,
            depthTest: true,
        })
        materialRef.current = material

        // Empty point cloud geometry (will be populated from server data)
        const geometry = new THREE.BufferGeometry()
        geometryRef.current = geometry

        const points = new THREE.Points(geometry, material)
        scene.add(points)
        pointCloudRef.current = points

        // Render loop
        const animate = () => {
            animFrameRef.current = requestAnimationFrame(animate)
            controls.update()

            // FPS counter
            fpsFramesRef.current++
            const now = performance.now()
            if (now - fpsTimeRef.current >= 1000) {
                onFps(fpsFramesRef.current)
                fpsFramesRef.current = 0
                fpsTimeRef.current = now
            }

            // Update time uniform
            material.uniforms.time.value = now * 0.001

            // Potree LOD: update visible nodes based on camera (~10 Hz, not every frame)
            if (potreeLoaderRef.current?.isLoaded && now - lastLodUpdateRef.current > 100) {
                lastLodUpdateRef.current = now
                potreeLoaderRef.current.updateVisibility()
                // Report visible points back to UI
                const visiblePts = potreeLoaderRef.current.getVisiblePointCount()
                if (visiblePts !== totalPointsRef.current) {
                    totalPointsRef.current = visiblePts
                    onPointCount(visiblePts)
                }
            }

            renderer.render(scene, camera)
        }
        animate()

        // Resize handler
        const handleResize = () => {
            if (!container) return
            const w = container.clientWidth
            const h = container.clientHeight
            camera.aspect = w / h
            camera.updateProjectionMatrix()
            renderer.setSize(w, h)
        }

        const resizeObserver = new ResizeObserver(handleResize)
        resizeObserver.observe(container)

        // Click + keyboard handlers for measurements
        const onCanvasClick = (e: MouseEvent) => handleMeasureClick(e)
        const onContextMenu = (e: MouseEvent) => {
            if (activeToolRef.current === 'measure-distance' || activeToolRef.current === 'measure-angle') {
                e.preventDefault()
                cancelPending()
            }
        }
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') cancelPending()
        }
        // Section box drag handlers
        const onSectionDown = (e: MouseEvent) => handleSectionMouseDown(e)
        const onSectionMove = (e: MouseEvent) => handleSectionMouseMove(e)
        const onSectionUp = () => handleSectionMouseUp()

        renderer.domElement.addEventListener('click', onCanvasClick)
        renderer.domElement.addEventListener('contextmenu', onContextMenu)
        renderer.domElement.addEventListener('mousedown', onSectionDown)
        renderer.domElement.addEventListener('mousemove', onSectionMove)
        renderer.domElement.addEventListener('mousemove', handleMeasureHover)
        renderer.domElement.addEventListener('mouseup', onSectionUp)
        window.addEventListener('keydown', onKeyDown)

        // Cleanup
        return () => {
            cancelAnimationFrame(animFrameRef.current)
            resizeObserver.disconnect()
            renderer.domElement.removeEventListener('click', onCanvasClick)
            renderer.domElement.removeEventListener('contextmenu', onContextMenu)
            renderer.domElement.removeEventListener('mousedown', onSectionDown)
            renderer.domElement.removeEventListener('mousemove', onSectionMove)
            renderer.domElement.removeEventListener('mousemove', handleMeasureHover)
            renderer.domElement.removeEventListener('mouseup', onSectionUp)
            window.removeEventListener('keydown', onKeyDown)
            controls.dispose()
            renderer.dispose()
            geometry.dispose()
            material.dispose()
            if (container.contains(renderer.domElement)) {
                container.removeChild(renderer.domElement)
            }
        }
    }, []) // Only run once on mount

    // Update point size when prop changes
    useEffect(() => {
        if (materialRef.current) {
            materialRef.current.uniforms.pointSize.value = pointSize
        }
    }, [pointSize])

    // Track activeSession in a ref for the WS onopen handler
    const activeSessionRef = useRef(activeSession)
    useEffect(() => { activeSessionRef.current = activeSession }, [activeSession])

    // Connect WebSocket with auto-reconnect
    useEffect(() => {
        let unmounted = false
        let reconnectTimer: ReturnType<typeof setTimeout> | null = null

        const connect = () => {
            if (unmounted) return
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
            const wsUrl = `${wsProtocol}//${window.location.host}/ws/viewer`
            console.log(`[Viewport] Connecting to ${wsUrl}`)

            const ws = new WebSocket(wsUrl)
            ws.binaryType = 'arraybuffer'
            wsRef.current = ws

            ws.onopen = () => {
                console.log('[Viewport] WebSocket connected')
                // If session was selected before WS was ready, load it now
                if (activeSessionRef.current) {
                    clearScene()
                    ws.send(JSON.stringify({
                        type: 'load_session',
                        session_id: activeSessionRef.current,
                    }))
                }
            }

            ws.onmessage = (event) => {
                if (event.data instanceof ArrayBuffer) {
                    handleBinaryData(event.data)
                } else {
                    try {
                        const msg = JSON.parse(event.data)
                        console.log('[Viewport] Message:', msg.type || msg)

                        // Handle 'cleared' — clean scene for next session
                        if (msg.type === 'cleared') {
                            clearScene()
                        }

                        handleJsonMessage(msg)
                    } catch {
                        // Ignore
                    }
                }
            }

            ws.onerror = (err) => {
                console.error('[Viewport] WebSocket error:', err)
            }

            ws.onclose = () => {
                console.log('[Viewport] WebSocket disconnected')
                // Only clear ref if this is still the active WS instance
                // (prevents React StrictMode cleanup from nulling the new WS)
                if (wsRef.current === ws) {
                    wsRef.current = null
                }
                // Auto-reconnect after 3 seconds
                if (!unmounted) {
                    reconnectTimer = setTimeout(connect, 3000)
                }
            }
        }

        connect()

        return () => {
            unmounted = true
            if (reconnectTimer) clearTimeout(reconnectTimer)
            wsRef.current?.close()
        }
    }, []) // Only setup once on mount

    // When activeSession changes, send load command on existing WS
    useEffect(() => {
        if (!activeSession) return
        const ws = wsRef.current
        if (!ws || ws.readyState !== WebSocket.OPEN) return

        // Clear existing scene before loading
        clearScene()

        ws.send(JSON.stringify({
            type: 'load_session',
            session_id: activeSession,
        }))
    }, [activeSession])

    // Clear geometry and OBBs
    const clearScene = useCallback(() => {
        // Dispose Potree loader if active
        if (potreeLoaderRef.current) {
            potreeLoaderRef.current.dispose()
            potreeLoaderRef.current = null
        }
        if (geometryRef.current) {
            geometryRef.current.dispose()
            const newGeometry = new THREE.BufferGeometry()
            geometryRef.current = newGeometry
            if (pointCloudRef.current) {
                pointCloudRef.current.geometry = newGeometry
            }
            totalPointsRef.current = 0
            onPointCount(0)
        }
        // Clear OBBs
        const group = obbGroupRef.current
        if (group) {
            while (group.children.length > 0) {
                const child = group.children[0]
                group.remove(child)
                if ((child as THREE.Mesh).geometry) (child as THREE.Mesh).geometry.dispose()
            }
            obbMapRef.current.clear()
        }
    }, [onPointCount])

    // Handle binary point cloud data from server
    const handleBinaryData = useCallback((buffer: ArrayBuffer) => {
        const floats = new Float32Array(buffer)
        const stride = 7 // x, y, z, r, g, b, classId
        const newPointCount = Math.floor(floats.length / stride)

        if (newPointCount === 0) return

        // Extract positions, colors, classIds
        const positions = new Float32Array(newPointCount * 3)
        const colors = new Float32Array(newPointCount * 3)
        const classIds = new Float32Array(newPointCount)

        for (let i = 0; i < newPointCount; i++) {
            const offset = i * stride
            positions[i * 3] = floats[offset]
            positions[i * 3 + 1] = floats[offset + 1]
            positions[i * 3 + 2] = floats[offset + 2]
            colors[i * 3] = floats[offset + 3]
            colors[i * 3 + 1] = floats[offset + 4]
            colors[i * 3 + 2] = floats[offset + 5]
            classIds[i] = floats[offset + 6]
        }

        // Create new geometry with accumulated data
        const geometry = geometryRef.current
        if (!geometry) return

        // If geometry already has positions, merge
        const existingPositions = geometry.getAttribute('position')
        if (existingPositions) {
            const oldPosArray = existingPositions.array as Float32Array
            const oldColorArray = (geometry.getAttribute('color')?.array || new Float32Array(0)) as Float32Array
            const oldClassArray = (geometry.getAttribute('classId')?.array || new Float32Array(0)) as Float32Array

            const mergedPos = new Float32Array(oldPosArray.length + positions.length)
            mergedPos.set(oldPosArray)
            mergedPos.set(positions, oldPosArray.length)

            const mergedCol = new Float32Array(oldColorArray.length + colors.length)
            mergedCol.set(oldColorArray)
            mergedCol.set(colors, oldColorArray.length)

            const mergedClass = new Float32Array(oldClassArray.length + classIds.length)
            mergedClass.set(oldClassArray)
            mergedClass.set(classIds, oldClassArray.length)

            geometry.setAttribute('position', new THREE.BufferAttribute(mergedPos, 3))
            geometry.setAttribute('color', new THREE.BufferAttribute(mergedCol, 3))
            geometry.setAttribute('classId', new THREE.BufferAttribute(mergedClass, 1))
        } else {
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
            geometry.setAttribute('classId', new THREE.BufferAttribute(classIds, 1))
        }

        geometry.computeBoundingSphere()
        totalPointsRef.current += newPointCount
        onPointCount(totalPointsRef.current)

        // Auto-center camera on first chunk
        if (totalPointsRef.current === newPointCount && controlsRef.current && cameraRef.current) {
            const center = new THREE.Vector3()
            geometry.boundingSphere?.center && center.copy(geometry.boundingSphere.center)
            controlsRef.current.target.copy(center)
            const radius = geometry.boundingSphere?.radius || 5
            cameraRef.current.position.set(
                center.x + radius * 1.5,
                center.y + radius,
                center.z + radius * 1.5
            )
            controlsRef.current.update()
        }
    }, [onPointCount])

    // Handle JSON messages from server (segmentation, status, potree_ready, etc)
    const handleJsonMessage = useCallback((msg: Record<string, unknown>) => {
        // ── Potree LOD: load octree via HTTP ──
        if (msg.type === 'potree_ready') {
            const url = msg.url as string
            const pts = msg.points as number
            const floorTransform = msg.floorTransform as number[] | undefined
            console.log(`[Viewport] Potree ready: ${pts?.toLocaleString()} points at ${url}`)
            if (onStatusMessage) onStatusMessage(`Loading LOD octree (${pts?.toLocaleString()} points)...`)

            // Create loader with existing material for section-box/segmentation compat
            const scene = sceneRef.current
            const camera = cameraRef.current
            const mat = materialRef.current
            if (!scene || !camera || !mat) return

            // Dispose previous loader
            if (potreeLoaderRef.current) {
                potreeLoaderRef.current.dispose()
            }

            const loader = new PotreeOctreeLoader(scene, camera, mat)
            potreeLoaderRef.current = loader

            loader.load(url).then((loadedPts) => {
                console.log(`[Viewport] Potree loaded: ${loadedPts.toLocaleString()} points`)
                if (onStatusMessage) onStatusMessage(`LOD octree loaded — ${pts?.toLocaleString()} total points`)
                onPointCount(loadedPts)

                // Apply floor alignment transform if provided
                if (floorTransform && floorTransform.length === 16) {
                    loader.setTransform(floorTransform)
                    // Store for camera centering only — OBBs are already
                    // floor-aligned on the backend so they don't need this
                    floorTransformRef.current = new THREE.Matrix4().fromArray(floorTransform)
                }

                // Auto-center camera on cloud bounding box
                const bbox = loader.getBoundingBox()
                if (bbox && cameraRef.current && controlsRef.current) {
                    // If transform was applied, transform the bbox too
                    if (floorTransform && floorTransform.length === 16) {
                        const m = new THREE.Matrix4().fromArray(floorTransform)
                        bbox.applyMatrix4(m)
                    }
                    const center = new THREE.Vector3()
                    bbox.getCenter(center)
                    const radius = bbox.getSize(new THREE.Vector3()).length() / 2
                    cameraRef.current.position.set(
                        center.x + radius * 1.5,
                        center.y + radius,
                        center.z + radius * 1.5
                    )
                    controlsRef.current.target.copy(center)
                    controlsRef.current.update()
                }
            }).catch((err) => {
                console.error('[Viewport] Potree load failed:', err)
                if (onStatusMessage) onStatusMessage(`Potree load error: ${err.message}`)
            })
            return // Don't process further
        }

        // v2.0 format: { instances: [{label, instance_id, color, total_points, obb, ...}] }
        if (Array.isArray(msg.instances)) {
            renderOBBs(msg.instances as Array<Record<string, unknown>>)
        }
        if (msg.type === 'status' || msg.type === 'progress') {
            const text = (msg.message || msg.status || '') as string
            if (text && onStatusMessage) onStatusMessage(text)
        }
        if (msg.type === 'pipeline_progress' && onPipelineProgress) {
            onPipelineProgress(msg)
        }
    }, [onStatusMessage, onSegments, onPipelineProgress, onPointCount])

    // Render OBB wireframe boxes for segmentation instances
    const renderOBBs = useCallback((instances: Array<Record<string, unknown>>) => {
        const group = obbGroupRef.current
        if (!group) return

        // Clear old OBBs
        while (group.children.length > 0) {
            const child = group.children[0]
            group.remove(child)
            if ((child as THREE.Mesh).geometry) (child as THREE.Mesh).geometry.dispose()
            if ((child as THREE.Mesh).material) {
                const mat = (child as THREE.Mesh).material
                if (Array.isArray(mat)) mat.forEach(m => m.dispose())
                else (mat as THREE.Material).dispose()
            }
        }
        obbMapRef.current.clear()

        const segmentList: SegmentInstance[] = []

        for (const inst of instances) {
            const obb = inst.obb as Record<string, unknown> | undefined
            const instId = (inst.instance_id || inst.id || 0) as number
            const label = (inst.label || 'object') as string
            const colorStr = (inst.color || '#00d4ff') as string
            const totalPoints = (inst.total_points || 0) as number
            const globalKey = (inst.global_id || `${label}_${instId}`) as string

            segmentList.push({
                key: globalKey,
                label: `${label} #${instId}`,
                color: colorStr,
                totalPoints,
                visible: true,
            })

            if (!obb) continue

            const center = obb.center as number[]
            const halfExtents = obb.half_extents as number[]
            const rotation = obb.rotation as number[][]
            if (!center || !halfExtents) continue

            // Color from hex string (e.g. '#FFFF00')
            const hexColor = new THREE.Color(colorStr)

            // Create box geometry
            const geom = new THREE.BoxGeometry(
                halfExtents[0] * 2,
                halfExtents[1] * 2,
                halfExtents[2] * 2
            )
            const edges = new THREE.EdgesGeometry(geom)
            const mat = new THREE.LineBasicMaterial({ color: hexColor, linewidth: 2 })
            const wireframe = new THREE.LineSegments(edges, mat)

            wireframe.position.set(center[0], center[1], center[2])

            // Apply rotation from 3x3 matrix
            if (rotation && rotation.length === 3) {
                const rotMatrix = new THREE.Matrix4()
                rotMatrix.set(
                    rotation[0][0], rotation[0][1], rotation[0][2], 0,
                    rotation[1][0], rotation[1][1], rotation[1][2], 0,
                    rotation[2][0], rotation[2][1], rotation[2][2], 0,
                    0, 0, 0, 1
                )
                wireframe.setRotationFromMatrix(rotMatrix)
            }

            group.add(wireframe)
            obbMapRef.current.set(globalKey, wireframe)
            geom.dispose()
        }

        // NOTE: OBBs are already in floor-aligned coordinates from the backend
        // (segmentation_pipeline.py applies floor transform to xyz_display before
        // computing OBBs). No additional transform needed here.

        console.log(`[Viewport] Rendered ${group.children.length} OBBs`)

        // Notify parent with segment list
        if (onSegments) onSegments(segmentList)
    }, [onSegments])

    return <div ref={containerRef} className="viewport-canvas" />
})

export default Viewport
