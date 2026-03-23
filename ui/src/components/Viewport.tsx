/**
 * STAC Build — 3D Viewport Component
 * Three.js-based point cloud renderer
 * Hernán Barreto — Ingerop IN3
 */
import { useRef, useEffect, useCallback, forwardRef, useImperativeHandle, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js'
import { PotreeOctreeLoader } from './PotreeLoader'

type Tool = 'navigate' | 'measure-distance' | 'measure-angle' | 'section-box' | 'align'

interface ViewportProps {
    pointSize: number
    confidenceThreshold: number
    activeSession: string | null
    activeTool: Tool
    showAxes?: boolean
    showGrid?: boolean
    pipelineRunning?: boolean
    onPointCount: (count: number) => void
    onFps: (fps: number) => void
    onStatusMessage?: (msg: string) => void
    onSegments?: (segments: SegmentInstance[]) => void
    onPipelineProgress?: (data: Record<string, unknown>) => void
    onBimLoaded?: (models: import('./IFCLoader').IFCLoadResult[]) => void
    onSabanaLoaded?: (pointCount: number) => void
    onHasConfidence?: (has: boolean) => void
    showCameraPoses?: boolean
    onHasCameraPoses?: (has: boolean) => void
}

export interface SegmentInstance {
    key: string
    id: number
    label: string
    color: string
    totalPoints: number
    visible: boolean
    excluded?: boolean
}

export interface ViewportHandle {
    sendCommand: (cmd: Record<string, unknown>) => void
    sendCommandPreserveCamera: (cmd: Record<string, unknown>) => void
    toggleOBB: (key: string, visible: boolean) => void
    toggleBIMVisibility: (meshNames: string[], visible: boolean) => void
    highlightBIMElement: (meshNames: string[]) => void
    addBIMGroup: (group: THREE.Group) => void
    removeBIMGroup: (filename: string) => void
    setBIMOpacity: (meshNames: string[], opacity: number) => void
    applyDeviationSurface: (sabanaData: Record<string, { positions: number[], colors: number[] }>, unmatchedKeys: string[]) => void
    applySabanaFromSaved: (positions: Float32Array, colors: Float32Array, nPoints: number) => void
    clearDeviationSurface: () => void
    applyRegistrationTransform: (transform: number[][]) => void
    clearMeasurements: () => void
    resetSectionBox: () => void
    resetCamera: () => void
    clearScene: () => void
    refreshSegmentOBBs: (sessionId: string) => void
    setOBBsVisible: (visible: boolean) => void
    setSegmentVisibility: (segId: number, visible: boolean) => void
}

// Vertex shader — matches FusionRenderer.js point size formula
const vertexShader = `
  attribute float classId;
  attribute float confidence;
  varying float vClassId;
  varying float vConfidence;
  varying float vSegVisible;
  varying vec3 vColor;
  varying vec3 vWorldPos;
  uniform float pointSize;
  uniform float uSegmentVisible[16];

  void main() {
    vClassId = classId;
    vConfidence = confidence;
    vColor = color;
    vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;

    // Segment visibility lookup (done here because classId attribute is exact)
    if (classId < 0.0) {
      vSegVisible = 1.0; // always visible (e.g. sábana)
    } else {
      int segIdx = int(clamp(classId, 0.0, 15.0));
      float sv = 1.0;
      if (segIdx == 0) sv = uSegmentVisible[0];
      else if (segIdx == 1) sv = uSegmentVisible[1];
      else if (segIdx == 2) sv = uSegmentVisible[2];
      else if (segIdx == 3) sv = uSegmentVisible[3];
      else if (segIdx == 4) sv = uSegmentVisible[4];
      else if (segIdx == 5) sv = uSegmentVisible[5];
      else if (segIdx == 6) sv = uSegmentVisible[6];
      else if (segIdx == 7) sv = uSegmentVisible[7];
      else if (segIdx == 8) sv = uSegmentVisible[8];
      else if (segIdx == 9) sv = uSegmentVisible[9];
      else if (segIdx == 10) sv = uSegmentVisible[10];
      else if (segIdx == 11) sv = uSegmentVisible[11];
      else if (segIdx == 12) sv = uSegmentVisible[12];
      else if (segIdx == 13) sv = uSegmentVisible[13];
      else if (segIdx == 14) sv = uSegmentVisible[14];
      else sv = uSegmentVisible[15];
      vSegVisible = sv;
    }

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;

    // Same formula as standalone viewer: pointScale / depth
    // Higher minimum (2.0) prevents vanishing at distance for sparser clouds
    float depth = -mvPosition.z;
    float size = pointSize / depth;
    gl_PointSize = clamp(size, 2.0, 30.0);
  }
`

// Fragment shader — circular points with EDL-like shading + section box clipping
const fragmentShader = `
  varying float vClassId;
  varying float vConfidence;
  varying float vSegVisible;
  varying vec3 vColor;
  varying vec3 vWorldPos;
  uniform float highlightIntensity;
  uniform float uOpacity;
  uniform float uConfidenceThreshold;
  uniform bool sectionBoxEnabled;
  uniform vec3 sectionBoxMin;
  uniform vec3 sectionBoxMax;

  void main() {
    // Segment visibility filter (computed in vertex shader for precision)
    if (vSegVisible < 0.5) discard;

    // Confidence filter: discard points below threshold
    // vConfidence defaults to 0.0 when attribute is absent; threshold 0.0 shows everything
    if (uConfidenceThreshold > 0.0 && vConfidence < uConfidenceThreshold) {
      discard;
    }

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
    
    gl_FragColor = vec4(finalColor, alpha * uOpacity);
  }
`

// ── Measurement helpers (outside component) ──
function createTextSprite(text: string, color: string = '#ffffff'): THREE.Sprite {
    const canvas = document.createElement('canvas')
    canvas.width = 128; canvas.height = 32
    const ctx = canvas.getContext('2d')!
    ctx.font = 'bold 22px sans-serif'
    ctx.fillStyle = 'rgba(0,0,0,0.7)'
    ctx.roundRect(0, 0, 128, 32, 4)
    ctx.fill()
    ctx.fillStyle = color
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(text, 64, 16)
    const texture = new THREE.CanvasTexture(canvas)
    const mat = new THREE.SpriteMaterial({ map: texture, depthTest: false, sizeAttenuation: true })
    const sprite = new THREE.Sprite(mat)
    sprite.scale.set(0.12, 0.03, 1)
    sprite.renderOrder = 1000
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

// ── frameCloud: isometric view framing the full point cloud extent ──
function frameCloud(bbox: THREE.Box3, cam: THREE.PerspectiveCamera, ctrl: { target: THREE.Vector3; update: () => void }) {
    const center = new THREE.Vector3()
    const size = new THREE.Vector3()
    bbox.getCenter(center)
    bbox.getSize(size)
    const maxDim = Math.max(size.x, size.y, size.z)
    // Distance to fit entire AABB — rotation inflation of the AABB
    // naturally provides visual margin so no extra factor needed
    const fov = cam.fov * (Math.PI / 180)
    const dist = (maxDim / 2) / Math.tan(fov / 2)
    // Isometric-like angle: 45° azimuth, 35° elevation (classic ISO)
    const phi = Math.PI / 180 * 35   // elevation
    const theta = Math.PI / 180 * 45 // azimuth
    cam.position.set(
        center.x + dist * Math.cos(phi) * Math.sin(theta),
        center.y + dist * Math.sin(phi),
        center.z + dist * Math.cos(phi) * Math.cos(theta)
    )
    ctrl.target.copy(center)
    ctrl.update()
}

// ── adaptGrid: resize grid to match cloud extents ──
function adaptGrid(
    bbox: THREE.Box3,
    scene: THREE.Scene,
    gridRef: React.MutableRefObject<THREE.GridHelper | null>,
    visible: boolean
) {
    if (gridRef.current) {
        scene.remove(gridRef.current)
        gridRef.current.geometry.dispose()
            ; (gridRef.current.material as THREE.Material).dispose()
    }
    const center = new THREE.Vector3()
    const size = new THREE.Vector3()
    bbox.getCenter(center)
    bbox.getSize(size)
    // Grid covers the XZ footprint of the cloud with slight margin
    const gridSize = Math.max(size.x, size.z) * 1.1
    const divisions = Math.max(10, Math.min(80, Math.round(gridSize / 0.5)))
    const grid = new THREE.GridHelper(gridSize, divisions, 0x252d3a, 0x1c2333)
    // Grid at Y=0 (floor level) — floor transform aligns floor to Y=0
    grid.position.set(center.x, 0, center.z)
    grid.visible = visible
    scene.add(grid)
    gridRef.current = grid
}

const Viewport = forwardRef<ViewportHandle, ViewportProps>(function Viewport(
    { pointSize, confidenceThreshold, activeSession, activeTool, showAxes = true, showGrid = true, pipelineRunning = false, onPointCount, onFps, onStatusMessage, onSegments, onPipelineProgress, onBimLoaded, onSabanaLoaded, onHasConfidence, showCameraPoses = true, onHasCameraPoses },
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
    const bimGroupRef = useRef<THREE.Group | null>(null)
    const sabanaGroupRef = useRef<THREE.Group | null>(null)
    const cameraGroupRef = useRef<THREE.Group | null>(null)
    const [camTooltip, setCamTooltip] = useState<{ x: number; y: number; frameName: string; sessionId: string } | null>(null)
    const [bimTooltip, setBimTooltip] = useState<{ x: number; y: number; type: string; name: string } | null>(null)
    const bimHoverThrottleRef = useRef(0)
    const [bimCtxMenu, setBimCtxMenu] = useState<{
        x: number; y: number; expressID: number; type: string; name: string; opacity: number
    } | null>(null)
    const camRaycasterRef = useRef(new THREE.Raycaster())

    // Measurement state
    const measureGroupRef = useRef<THREE.Group | null>(null)
    const pendingPointsRef = useRef<THREE.Vector3[]>([])
    const pendingMarkersRef = useRef<THREE.Mesh[]>([])
    const pendingLineRef = useRef<THREE.Line | null>(null)
    const livePreviewRef = useRef<THREE.Group | null>(null)
    const measurementsRef = useRef<Measurement[]>([])
    const raycasterRef = useRef(new THREE.Raycaster())
    const activeToolRef = useRef(activeTool)
    const hoverHighlightRef = useRef<THREE.Group | null>(null)
    const potreeLoaderRef = useRef<PotreeOctreeLoader | null>(null)
    const lastLodUpdateRef = useRef(0)
    const sabanaLoadIdRef = useRef(0)  // monotonic counter to discard stale sábana loads
    const sessionFramedRef = useRef<string | null>(null)  // track which session has been framed
    const preserveCameraRef = useRef<{ pos: THREE.Vector3; target: THREE.Vector3 } | null>(null)
    const floorTransformRef = useRef<THREE.Matrix4 | null>(null)
    const gridRef = useRef<THREE.GridHelper | null>(null)
    const cloudBBoxRef = useRef<THREE.Box3 | null>(null)
    const axesRef = useRef<THREE.Group | null>(null)
    const showGridRef = useRef(showGrid)
    const showCameraPosesRef = useRef(showCameraPoses)
    const pipelineRunningRef = useRef(pipelineRunning)

    // Alignment gizmo state
    const transformControlsRef = useRef<TransformControls | null>(null)
    const alignPivotRef = useRef<THREE.Group | null>(null)
    const alignSavedRef = useRef(false)  // tracks if alignment was saved during this session
    const [alignMode, setAlignMode] = useState<'translate' | 'rotate'>('rotate')
    const [alignDirty, setAlignDirty] = useState(false)

    // Keep activeToolRef in sync with prop
    useEffect(() => { activeToolRef.current = activeTool }, [activeTool])

    // Toggle grid and axes visibility
    useEffect(() => { if (gridRef.current) gridRef.current.visible = showGrid; showGridRef.current = showGrid }, [showGrid])
    useEffect(() => { pipelineRunningRef.current = pipelineRunning }, [pipelineRunning])
    useEffect(() => { if (axesRef.current) axesRef.current.visible = showAxes }, [showAxes])

    // Toggle OrbitControls left-button based on active tool
    useEffect(() => {
        const controls = controlsRef.current
        if (!controls) return
        if (activeTool === 'measure-distance' || activeTool === 'measure-angle' || activeTool === 'section-box' || activeTool === 'align') {
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
        // Remove live preview
        if (livePreviewRef.current) {
            group.remove(livePreviewRef.current)
            livePreviewRef.current.traverse(c => {
                if ((c as any).geometry) (c as any).geometry.dispose()
                if ((c as any).material) {
                    const m = (c as any).material
                    if (m.map) m.map.dispose()
                    m.dispose()
                }
            })
            livePreviewRef.current = null
        }
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

        // Reset pending + clean up live preview
        pendingPointsRef.current = []
        pendingMarkersRef.current = []
        pendingLineRef.current = null
        if (livePreviewRef.current) {
            group.remove(livePreviewRef.current)
            livePreviewRef.current.traverse(c => {
                if ((c as any).geometry) (c as any).geometry.dispose()
                if ((c as any).material) { const m = (c as any).material; if (m.map) m.map.dispose(); m.dispose() }
            })
            livePreviewRef.current = null
        }

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
        if (livePreviewRef.current) {
            group.remove(livePreviewRef.current)
            livePreviewRef.current.traverse(c => {
                if ((c as any).geometry) (c as any).geometry.dispose()
                if ((c as any).material) { const m = (c as any).material; if (m.map) m.map.dispose(); m.dispose() }
            })
            livePreviewRef.current = null
        }

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

        // Live preview line + distance from first pending point to hover
        const pending = pendingPointsRef.current
        const group = measureGroupRef.current
        if (pending.length >= 1 && group) {
            // Remove old preview
            if (livePreviewRef.current) {
                group.remove(livePreviewRef.current)
                livePreviewRef.current.traverse(c => {
                    if ((c as any).geometry) (c as any).geometry.dispose()
                    if ((c as any).material) {
                        const m = (c as any).material
                        if (m.map) m.map.dispose()
                        m.dispose()
                    }
                })
            }

            const previewGroup = new THREE.Group()
            const lastPt = pending[pending.length - 1]
            const lineColor = tool === 'measure-distance' ? 0x00ff88 : 0xffaa00

            // Dashed line
            const lineGeom = new THREE.BufferGeometry().setFromPoints([lastPt, hitPoint])
            const lineMat = new THREE.LineDashedMaterial({
                color: lineColor, dashSize: 0.03, gapSize: 0.02,
                transparent: true, opacity: 0.7, depthTest: false
            })
            const dashedLine = new THREE.Line(lineGeom, lineMat)
            dashedLine.computeLineDistances()
            dashedLine.renderOrder = 999
            previewGroup.add(dashedLine)

            // Distance label sprite
            const dist = lastPt.distanceTo(hitPoint)
            const distStr = dist < 1 ? `${(dist * 100).toFixed(1)} cm` : `${dist.toFixed(3)} m`
            const lCanvas = document.createElement('canvas')
            lCanvas.width = 128; lCanvas.height = 32
            const lCtx = lCanvas.getContext('2d')!
            lCtx.font = 'bold 22px sans-serif'
            lCtx.fillStyle = 'rgba(0,0,0,0.7)'
            lCtx.roundRect(0, 0, 128, 32, 4)
            lCtx.fill()
            lCtx.fillStyle = '#ffffff'
            lCtx.textAlign = 'center'
            lCtx.textBaseline = 'middle'
            lCtx.fillText(distStr, 64, 16)
            const lTex = new THREE.CanvasTexture(lCanvas)
            const lMat = new THREE.SpriteMaterial({ map: lTex, depthTest: false })
            const lSprite = new THREE.Sprite(lMat)
            const mid = new THREE.Vector3().lerpVectors(lastPt, hitPoint, 0.5)
            lSprite.position.copy(mid)
            lSprite.position.y += 0.04
            lSprite.scale.set(0.12, 0.03, 1)
            lSprite.renderOrder = 1000
            previewGroup.add(lSprite)

            group.add(previewGroup)
            livePreviewRef.current = previewGroup
        }
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

    // ── Alignment gizmo effect ──────────────────────────────────
    useEffect(() => {
        const scene = sceneRef.current
        const camera = cameraRef.current
        const renderer = rendererRef.current
        const controls = controlsRef.current
        if (!scene || !camera || !renderer || !controls) return

        if (activeTool === 'align') {
            // Create a pivot at the octree center
            const loader = potreeLoaderRef.current
            const obbGroup = obbGroupRef.current
            if (!loader) return

            const pivot = new THREE.Group()
            pivot.name = 'alignPivot'

            // Compute cloud center for pivot position
            const bbox = loader.getBoundingBox()
            if (bbox) {
                // If there's a floor transform, apply it to bbox
                if (floorTransformRef.current) {
                    bbox.applyMatrix4(floorTransformRef.current)
                }
                const center = new THREE.Vector3()
                bbox.getCenter(center)
                pivot.position.copy(center)
            }

            scene.add(pivot)
            alignPivotRef.current = pivot

            // Create TransformControls
            const tc = new TransformControls(camera, renderer.domElement)
            tc.attach(pivot)
            tc.setMode('rotate')
            tc.setSize(1.2)
            scene.add(tc.getHelper())
            transformControlsRef.current = tc

            // Disable orbit while dragging gizmo
            tc.addEventListener('dragging-changed', (event: any) => {
                controls.enabled = !event.value
            })

            // ── Simulated parenting: compute relative matrices ──
            // Instead of complex pivot delta math, we compute how each group
            // is positioned RELATIVE to the pivot. Then on each objectChange,
            // we just apply: groupMatrix = pivotWorldMatrix * relativeMatrix
            // This is what Three.js parenting does, without actually reparenting.
            const pivotInitialM = new THREE.Matrix4().compose(pivot.position, pivot.quaternion, pivot.scale)
            const pivotInitialInv = pivotInitialM.clone().invert()

            const octreeGroup = loader.getOctreeGroup()
            const octreeInitialM = floorTransformRef.current ? floorTransformRef.current.clone() : new THREE.Matrix4()
            const octreeRelative = pivotInitialInv.clone().multiply(octreeInitialM)

            const obbInitialM = obbGroup ? obbGroup.matrix.clone() : new THREE.Matrix4()
            const obbRelative = pivotInitialInv.clone().multiply(obbInitialM)

            // Track changes
            tc.addEventListener('objectChange', () => {
                setAlignDirty(true)
                // Compute current pivot world matrix
                const pivotWorld = new THREE.Matrix4().compose(pivot.position, pivot.quaternion, pivot.scale)

                // Apply to octreeGroup: pivotWorld * octreeRelative
                if (octreeGroup) {
                    octreeGroup.matrix.copy(pivotWorld.clone().multiply(octreeRelative))
                    octreeGroup.matrixAutoUpdate = false
                    octreeGroup.matrixWorldNeedsUpdate = true
                }
                // Apply to obbGroup: pivotWorld * obbRelative
                if (obbGroup) {
                    obbGroup.matrix.copy(pivotWorld.clone().multiply(obbRelative))
                    obbGroup.matrixAutoUpdate = false
                    obbGroup.matrixWorldNeedsUpdate = true
                }
            })

            setAlignDirty(false)
            setAlignMode('rotate')
            alignSavedRef.current = false
            if (onStatusMessage) onStatusMessage('⛶ Align mode: drag gizmo to rotate/translate')

            return () => {
                // Cleanup gizmo
                scene.remove(tc.getHelper())
                tc.detach()
                tc.dispose()
                transformControlsRef.current = null

                // Restore octreeGroup from floorTransformRef (which is updated on save)
                const octreeGroup = loader.getOctreeGroup()
                if (octreeGroup) {
                    if (floorTransformRef.current) {
                        // Re-enable auto matrix before setTransform (objectChange set it to false)
                        octreeGroup.matrixAutoUpdate = true
                        loader.setTransform(floorTransformRef.current.toArray())
                    } else {
                        octreeGroup.matrix.identity()
                        octreeGroup.matrixAutoUpdate = true
                    }
                    octreeGroup.matrixWorldNeedsUpdate = true
                }
                // If alignment was saved, KEEP the OBB group transform
                // (OBBs are in old display-space and need the delta to match new cloud)
                // If NOT saved (user cancelled), reset OBBs to identity
                if (obbGroup && !alignSavedRef.current) {
                    obbGroup.matrix.identity()
                    obbGroup.matrixAutoUpdate = true
                    obbGroup.matrixWorldNeedsUpdate = true
                }
                scene.remove(pivot)
                alignPivotRef.current = null
                setAlignDirty(false)
                controls.enabled = true
            }
        }
    }, [activeTool, onStatusMessage])

    // Sync gizmo mode with alignMode state
    useEffect(() => {
        if (transformControlsRef.current) {
            transformControlsRef.current.setMode(alignMode)
        }
    }, [alignMode])

    // Save alignment function — NO session reload, instant
    const saveAlignment = useCallback(async () => {
        const loader = potreeLoaderRef.current
        if (!loader) return

        // Read the octreeGroup's current matrix — already has correct composed transform
        const octreeGroup = loader.getOctreeGroup()
        octreeGroup.updateMatrixWorld(true)
        const composedM = octreeGroup.matrixWorld.clone()
        const arr = composedM.toArray() // column-major

        try {
            if (onStatusMessage) onStatusMessage('💾 Saving alignment...')
            const res = await fetch(`/api/sessions/${activeSession}/alignment`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ transform: arr })
            })
            if (!res.ok) throw new Error('Failed to save alignment')

            // Update stored floor transform to the new composed matrix
            floorTransformRef.current = composedM.clone()

            // Apply permanently to Potree loader (decompose into pos/quat/scale)
            loader.setTransform(arr)

            // Reset gizmo pivot to new cloud center
            const pivot = alignPivotRef.current
            if (pivot) {
                const bbox = loader.getBoundingBox()
                if (bbox) {
                    bbox.applyMatrix4(composedM)
                    const center = new THREE.Vector3()
                    bbox.getCenter(center)
                    pivot.position.copy(center)
                }
                pivot.quaternion.identity()
                pivot.scale.set(1, 1, 1)
            }

            alignSavedRef.current = true
            setAlignDirty(false)
            if (onStatusMessage) onStatusMessage('✅ Alignment saved! Reload session to see updated OBBs.')
        } catch (e: any) {
            if (onStatusMessage) onStatusMessage(`Error saving alignment: ${e.message}`)
        }
    }, [activeSession, onStatusMessage])

    // Expose sendCommand + toggleOBB + clearMeasurements + resetSectionBox to parent via ref
    useImperativeHandle(ref, () => ({
        sendCommand: (cmd: Record<string, unknown>) => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify(cmd))
            }
        },
        sendCommandPreserveCamera: (cmd: Record<string, unknown>) => {
            // Save current camera state before reload
            if (cameraRef.current && controlsRef.current) {
                preserveCameraRef.current = {
                    pos: cameraRef.current.position.clone(),
                    target: controlsRef.current.target.clone(),
                }
            }
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify(cmd))
            }
        },
        toggleOBB: (key: string, visible: boolean) => {
            const mesh = obbMapRef.current.get(key)
            if (mesh) mesh.visible = visible
        },
        setOBBsVisible: (visible: boolean) => {
            const group = obbGroupRef.current
            if (group) group.visible = visible
        },
        setSegmentVisibility: (segId: number, visible: boolean) => {
            const mat = materialRef.current
            if (!mat) return
            const idx = Math.max(0, Math.min(segId, 15))
            const oldArr = mat.uniforms.uSegmentVisible.value as number[]
            const newArr = [...oldArr]
            newArr[idx] = visible ? 1.0 : 0.0
            mat.uniforms.uSegmentVisible.value = newArr
            mat.uniformsNeedUpdate = true
        },
        toggleBIMVisibility: (meshNames: string[], visible: boolean) => {
            const bimGroup = bimGroupRef.current
            if (!bimGroup) return
            bimGroup.traverse((child) => {
                if (meshNames.includes(child.name)) {
                    child.visible = visible
                }
            })
        },
        highlightBIMElement: (meshNames: string[]) => {
            const bimGroup = bimGroupRef.current
            if (!bimGroup) return
            // Reset all BIM materials, then highlight selected
            bimGroup.traverse((child) => {
                if (child instanceof THREE.Mesh && child.userData?.expressID) {
                    const mat = child.material as any
                    if (mat._originalEmissive !== undefined) {
                        mat.emissive.setHex(mat._originalEmissive)
                    }
                }
            })
            bimGroup.traverse((child) => {
                if (child instanceof THREE.Mesh && meshNames.includes(child.name)) {
                    const mat = child.material as any
                    if (mat._originalEmissive === undefined) {
                        mat._originalEmissive = mat.emissive.getHex()
                    }
                    mat.emissive.setHex(0x335599)
                }
            })
        },
        addBIMGroup: (group: THREE.Group) => {
            const bimGroup = bimGroupRef.current
            if (bimGroup) bimGroup.add(group)
        },
        removeBIMGroup: (filename: string) => {
            const bimGroup = bimGroupRef.current
            if (!bimGroup) return
            const groupName = `ifc_${filename}`
            const toRemove = bimGroup.children.filter(c => c.name === groupName)
            for (const child of toRemove) {
                child.traverse((obj) => {
                    if (obj instanceof THREE.Mesh) {
                        obj.geometry?.dispose()
                        if (obj.material) {
                            if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose())
                            else obj.material.dispose()
                        }
                    }
                })
                bimGroup.remove(child)
            }
            console.log(`[Viewport] Removed BIM group: ${groupName} (${toRemove.length} groups)`)
        },
        setBIMOpacity: (meshNames: string[], opacity: number) => {
            const bimGroup = bimGroupRef.current
            if (!bimGroup) return
            bimGroup.traverse((child) => {
                if (child instanceof THREE.Mesh && meshNames.includes(child.name)) {
                    const mat = child.material as THREE.MeshStandardMaterial
                    mat.opacity = opacity
                    mat.transparent = opacity < 1.0
                    mat.needsUpdate = true
                }
            })
        },
        applyDeviationSurface: (sabanaData: Record<string, { positions: number[], colors: number[] }>, unmatchedKeys: string[]) => {
            const scene = sceneRef.current
            const bimGroup = bimGroupRef.current
            if (!scene || !bimGroup) return

            // Clean up previous sábana
            if (sabanaGroupRef.current) {
                scene.remove(sabanaGroupRef.current)
                sabanaGroupRef.current.traverse((c) => {
                    if (c instanceof THREE.Points) {
                        c.geometry.dispose()
                            ; (c.material as THREE.Material).dispose()
                    }
                })
            }

            const sabanaGroup = new THREE.Group()
            sabanaGroup.name = 'sabana'
            sabanaGroupRef.current = sabanaGroup

            // ── Evaluated elements: render as colored point cloud ──
            let totalPoints = 0
            for (const [key, data] of Object.entries(sabanaData)) {
                const nPts = Math.floor(data.positions.length / 3)
                if (nPts === 0) continue

                const posArr = new Float32Array(data.positions)
                const colArr = new Float32Array(nPts * 3)

                // Convert RGBA → RGB for PointsMaterial (alpha is uniform)
                for (let i = 0; i < nPts; i++) {
                    colArr[i * 3] = data.colors[i * 4]
                    colArr[i * 3 + 1] = data.colors[i * 4 + 1]
                    colArr[i * 3 + 2] = data.colors[i * 4 + 2]
                }

                const geom = new THREE.BufferGeometry()
                geom.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
                geom.setAttribute('color', new THREE.BufferAttribute(colArr, 3))

                const mat = new THREE.PointsMaterial({
                    size: 0.003,  // 3mm — matches scan spacing
                    vertexColors: true,
                    sizeAttenuation: true,
                    depthWrite: true,
                })

                const points = new THREE.Points(geom, mat)
                points.name = `sabana_${key}`
                sabanaGroup.add(points)
                totalPoints += nPts
            }

            // ── Make ALL BIM meshes semi-transparent so sábana stands out ──
            bimGroup.traverse((child) => {
                if (!(child instanceof THREE.Mesh)) return
                if (!child.userData._originalMaterial) {
                    child.userData._originalMaterial = child.material
                }
                const meshName = child.name || ''
                const ifcName = child.userData?.ifc_name || ''
                // Check if this is an unmatched element → more transparent
                let isUnmatched = false
                for (const key of unmatchedKeys) {
                    if (ifcName.endsWith(':' + key) || ifcName === key ||
                        meshName.includes(':' + key + '_') || meshName.includes(':' + key)) {
                        isUnmatched = true
                        break
                    }
                }
                child.material = new THREE.MeshBasicMaterial({
                    color: isUnmatched ? 0x808080 : 0xaaaaaa,
                    transparent: true,
                    opacity: isUnmatched ? 0.15 : 0.20,
                    side: THREE.DoubleSide,
                    depthWrite: false,
                })
            })

            // ── Dim Potree scan cloud via shared material uniform ──
            const sharedMat = materialRef.current
            if (sharedMat) {
                sharedMat.uniforms.uOpacity.value = 0.15
            }

            scene.add(sabanaGroup)
            console.log(`[Viewport] Sábana: ${totalPoints} points, ${Object.keys(sabanaData).length} evaluated, ${unmatchedKeys.length} unmatched`)
        },
        applySabanaFromSaved: (positions: Float32Array, colors: Float32Array, nPoints: number) => {
            const scene = sceneRef.current
            if (!scene) return

            // Clear previous
            if (sabanaGroupRef.current) {
                scene.remove(sabanaGroupRef.current)
                sabanaGroupRef.current.traverse((c) => {
                    if (c instanceof THREE.Points) {
                        c.geometry.dispose()
                            ; (c.material as THREE.Material).dispose()
                    }
                })
            }

            const sabanaGroup = new THREE.Group()
            sabanaGroup.name = 'sabana-group'
            sabanaGroupRef.current = sabanaGroup

            // Build geometry from flat arrays
            const posArr = positions
            const colArr = colors
            const geo = new THREE.BufferGeometry()
            geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
            geo.setAttribute('color', new THREE.BufferAttribute(colArr, 4))

            const mat = new THREE.PointsMaterial({
                size: 0.008,
                vertexColors: true,
                transparent: true,
                depthWrite: false,
                sizeAttenuation: true,
            })
            // Use vertex alpha
            mat.onBeforeCompile = (shader) => {
                shader.fragmentShader = shader.fragmentShader.replace(
                    'vec4 diffuseColor = vec4( diffuse, opacity );',
                    'vec4 diffuseColor = vec4( diffuse, opacity * vColor.a );'
                )
            }

            const pts = new THREE.Points(geo, mat)
            sabanaGroup.add(pts)

            // ── Dim BIM meshes ──
            const bimGroup = bimGroupRef.current
            if (bimGroup) {
                bimGroup.traverse((child) => {
                    if (!(child instanceof THREE.Mesh)) return
                    if (!child.userData._originalMaterial) {
                        child.userData._originalMaterial = child.material
                    }
                    child.material = new THREE.MeshBasicMaterial({
                        color: 0x888888,
                        transparent: true,
                        opacity: 0.20,
                        side: THREE.DoubleSide,
                        depthWrite: false,
                    })
                })
            }

            // ── Dim Potree scan cloud ──
            const sharedMat = materialRef.current
            if (sharedMat) {
                sharedMat.uniforms.uOpacity.value = 0.15
            }

            scene.add(sabanaGroup)
            console.log(`[Viewport] Sábana loaded from saved: ${nPoints} points`)
        },
        clearDeviationSurface: () => {
            const scene = sceneRef.current
            // Remove sábana point cloud
            if (sabanaGroupRef.current && scene) {
                scene.remove(sabanaGroupRef.current)
                sabanaGroupRef.current.traverse((c) => {
                    if (c instanceof THREE.Points) {
                        c.geometry.dispose()
                            ; (c.material as THREE.Material).dispose()
                    }
                })
                sabanaGroupRef.current = null
            }
            // Restore BIM mesh materials
            const bimGroup = bimGroupRef.current
            if (bimGroup) {
                bimGroup.traverse((child) => {
                    if (!(child instanceof THREE.Mesh)) return
                    if (child.userData._originalMaterial) {
                        if (child.material !== child.userData._originalMaterial) {
                            (child.material as THREE.Material).dispose()
                        }
                        child.material = child.userData._originalMaterial
                        delete child.userData._originalMaterial
                    }
                })
            }
            // Restore Potree scan cloud opacity
            const sharedMat = materialRef.current
            if (sharedMat) {
                sharedMat.uniforms.uOpacity.value = 1.0
            }
        },
        applyRegistrationTransform: (transform: number[][]) => {
            // Convert row-major 4x4 from backend to Three.js column-major Matrix4
            const m = new THREE.Matrix4()
            m.set(
                transform[0][0], transform[0][1], transform[0][2], transform[0][3],
                transform[1][0], transform[1][1], transform[1][2], transform[1][3],
                transform[2][0], transform[2][1], transform[2][2], transform[2][3],
                transform[3][0], transform[3][1], transform[3][2], transform[3][3],
            )
            const loader = potreeLoaderRef.current
            if (loader) {
                // Compute delta for OBBs: OBBs are in viewer space (floor_transform applied)
                // delta = T_registration × T_floor_inverse
                const currentTransform = floorTransformRef.current
                if (currentTransform) {
                    const invFloor = currentTransform.clone().invert()
                    const delta = m.clone().multiply(invFloor)
                    const obbGroup = obbGroupRef.current
                    if (obbGroup) {
                        obbGroup.matrix.copy(delta)
                        obbGroup.matrixAutoUpdate = false
                        obbGroup.matrixWorldNeedsUpdate = true
                    }
                }
                // Apply full transform to point cloud
                loader.setTransform(m.toArray())
                floorTransformRef.current = m.clone()
                console.log('[Viewport] Registration transform applied to point cloud')
            }
        },
        clearMeasurements: clearAllMeasurements,
        resetSectionBox: destroySectionBox,
        resetCamera: () => {
            const cam = cameraRef.current
            const ctrl = controlsRef.current
            if (!cam || !ctrl) return
            const bbox = cloudBBoxRef.current
            if (bbox && !bbox.isEmpty()) {
                frameCloud(bbox, cam, ctrl)
            } else {
                cam.position.set(5, 5, 5)
                cam.lookAt(0, 0, 0)
                ctrl.target.set(0, 0, 0)
                ctrl.update()
            }
        },
        clearScene,
        refreshSegmentOBBs: async (sessionId: string) => {
            try {
                const res = await fetch(`/api/sessions/${sessionId}/segmentation`)
                if (res.ok) {
                    const data = await res.json()
                    if (Array.isArray(data.instances)) {
                        renderOBBs(data.instances)
                        console.log(`[Viewport] Refreshed ${data.instances.length} segment OBBs`)
                    }
                }
            } catch (err) {
                console.error('[Viewport] Failed to refresh segment OBBs:', err)
            }
        },
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
        // Very subtle fog — only noticeable at 200+ meters, never obscures close-up detail
        scene.fog = new THREE.FogExp2(0x0d1117, 0.0003)
        sceneRef.current = scene

        // Lights for BIM/mesh rendering (PBR materials need lights)
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
        scene.add(ambientLight)
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8)
        dirLight.position.set(10, 20, 10)
        scene.add(dirLight)
        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3)
        dirLight2.position.set(-10, -5, -10)
        scene.add(dirLight2)

        // OBB group for segmentation bounding boxes
        const obbGroup = new THREE.Group()
        obbGroup.name = 'obbGroup'
        scene.add(obbGroup)
        obbGroupRef.current = obbGroup

        // BIM group for IFC models
        const bimGroup = new THREE.Group()
        bimGroup.name = 'bimGroup'
        scene.add(bimGroup)
        bimGroupRef.current = bimGroup

        // Camera poses group
        const cameraGroup = new THREE.Group()
        cameraGroup.name = 'cameraGroup'
        scene.add(cameraGroup)
        cameraGroupRef.current = cameraGroup

        // Measurement overlay group
        const measureGroup = new THREE.Group()
        measureGroup.name = 'measureGroup'
        scene.add(measureGroup)
        measureGroupRef.current = measureGroup

        // Grid helper (will be replaced when cloud loads)
        const gridHelper = new THREE.GridHelper(20, 40, 0x252d3a, 0x1c2333)
        gridHelper.visible = showGrid
        scene.add(gridHelper)
        gridRef.current = gridHelper

        // Axes helper with XYZ labels
        const axesGroup = new THREE.Group()
        const axesHelper = new THREE.AxesHelper(2)
        axesGroup.add(axesHelper)

        // Create text sprite labels for X, Y, Z
        const makeLabel = (text: string, color: string, pos: [number, number, number]) => {
            const canvas = document.createElement('canvas')
            canvas.width = 64; canvas.height = 64
            const ctx = canvas.getContext('2d')!
            ctx.font = 'bold 48px sans-serif'
            ctx.fillStyle = color
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            ctx.fillText(text, 32, 32)
            const tex = new THREE.CanvasTexture(canvas)
            const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false })
            const sprite = new THREE.Sprite(mat)
            sprite.position.set(...pos)
            sprite.scale.set(0.15, 0.15, 0.15)
            return sprite
        }
        axesGroup.add(makeLabel('X', '#ff4444', [2.15, 0, 0]))
        axesGroup.add(makeLabel('Y', '#44ff44', [0, 2.15, 0]))
        axesGroup.add(makeLabel('Z', '#4488ff', [0, 0, 2.15]))
        scene.add(axesGroup)
        axesGroup.visible = showAxes
        axesRef.current = axesGroup

        // Camera
        const camera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.001,   // 1mm near plane — allows extreme close-ups
            10000    // 10km far plane — handles very large sites
        )
        camera.position.set(5, 5, 5)
        camera.lookAt(0, 0, 0)
        cameraRef.current = camera

        // Controls — using ONLY native OrbitControls features.
        // CRITICAL: OrbitControls.update() always overwrites camera.position
        // from internal spherical state. Any external manipulation = bounce.
        // So we use the built-in zoomToCursor instead of custom handlers.
        const controls = new OrbitControls(camera, renderer.domElement)
        controls.enableDamping = true
        controls.dampingFactor = 0.1
        controls.screenSpacePanning = true
        controls.minDistance = 0
        controls.maxDistance = 5000
        controls.maxPolarAngle = Math.PI
        controls.enableZoom = true
        controls.zoomSpeed = 8.0          // Fast zoom for large clouds
        controls.zoomToCursor = true      // Zoom toward mouse pointer + auto-reposition target
        controlsRef.current = controls

        // Shader material for point cloud
        const material = new THREE.ShaderMaterial({
            vertexShader,
            fragmentShader,
            uniforms: {
                pointSize: { value: pointSize },
                highlightIntensity: { value: 0.5 },
                uOpacity: { value: 1.0 },
                uConfidenceThreshold: { value: 0.0 },
                uSegmentVisible: { value: new Array(16).fill(1.0) },
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

            // Re-anchor target: when zoom brings camera very close to target,
            // push target forward to maintain minimum orbit radius.
            // This prevents logarithmic zoom from stalling at close range.
            // Safe because we modify target AFTER update() wrote camera.position.
            const orbitDist = camera.position.distanceTo(controls.target)
            if (orbitDist < 0.5) {
                const forward = new THREE.Vector3(0, 0, -1)
                    .transformDirection(camera.matrixWorld)
                controls.target.copy(camera.position).addScaledVector(forward, 0.5)
            }

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

    // Update confidence threshold when prop changes
    useEffect(() => {
        if (materialRef.current) {
            materialRef.current.uniforms.uConfidenceThreshold.value = confidenceThreshold
        }
        // Hide point cloud at max confidence to show only voxel mesh + wireframe
        const hideCloud = confidenceThreshold >= 1.0
        if (pointCloudRef.current) {
            pointCloudRef.current.visible = !hideCloud
        }
        // Also hide Potree octree
        const potreeGroup = sceneRef.current?.getObjectByName('potree-octree')
        if (potreeGroup) {
            potreeGroup.visible = !hideCloud
        }
    }, [confidenceThreshold])

    // Track activeSession in a ref for the WS onopen handler
    const activeSessionRef = useRef(activeSession)
    useEffect(() => {
        activeSessionRef.current = activeSession
        sessionFramedRef.current = null  // reset so new session gets framed
    }, [activeSession])

    // Connect WebSocket with auto-reconnect
    useEffect(() => {
        let unmounted = false
        let reconnectTimer: ReturnType<typeof setTimeout> | null = null

        const connect = () => {
            if (unmounted) return
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
            const wsUrl = `${wsProtocol}//${window.location.host}/ws/viewer`
            // console.log(`[Viewport] Connecting to ${wsUrl}`)

            const ws = new WebSocket(wsUrl)
            ws.binaryType = 'arraybuffer'
            wsRef.current = ws

            ws.onopen = () => {
                // console.log('[Viewport] WebSocket connected')
                // If session was selected before WS was ready, load it now
                // BUT: skip reload if pipeline is running (no PLY data yet, causes grid flicker)
                if (activeSessionRef.current && !pipelineRunningRef.current) {
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
                        // console.log('[Viewport] Message:', msg.type || msg)

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
                // console.log('[Viewport] WebSocket disconnected')
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

    // Camera pose hover raycasting
    useEffect(() => {
        const container = containerRef.current
        if (!container) return
        const handleMouseMove = (e: MouseEvent) => {
            const camGroup = cameraGroupRef.current
            const camera = cameraRef.current
            if (!camGroup || !camera || camGroup.children.length === 0 || !showCameraPosesRef.current) {
                setCamTooltip(null)
                return
            }
            const rect = container.getBoundingClientRect()
            const mouse = new THREE.Vector2(
                ((e.clientX - rect.left) / rect.width) * 2 - 1,
                -((e.clientY - rect.top) / rect.height) * 2 + 1
            )
            const rc = camRaycasterRef.current
            rc.setFromCamera(mouse, camera)
            // Only raycast against spheres (Mesh children, not ArrowHelpers)
            const spheres = camGroup.children.filter(c => (c as THREE.Mesh).isMesh && c.userData.isCamPose)
            const hits = rc.intersectObjects(spheres, false)
            if (hits.length > 0) {
                const hit = hits[0].object
                setCamTooltip({
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top,
                    frameName: hit.userData.frameName,
                    sessionId: hit.userData.sessionId,
                })
            } else {
                setCamTooltip(null)
            }

            // BIM element hover raycast (throttled)
            const now = performance.now()
            if (now - bimHoverThrottleRef.current > 80) {
                bimHoverThrottleRef.current = now
                const bimGroup = bimGroupRef.current
                if (bimGroup && bimGroup.children.length > 0 && bimGroup.visible) {
                    const bimHits = rc.intersectObjects(bimGroup.children, true)
                      .filter(h => h.object.visible)
                    if (bimHits.length > 0) {
                        const bimHit = bimHits[0].object
                        const ifcType = bimHit.userData?.ifc_type || ''
                        const ifcName = bimHit.userData?.ifc_name || ''
                        if (ifcType || ifcName) {
                            setBimTooltip({
                                x: e.clientX - rect.left,
                                y: e.clientY - rect.top,
                                type: ifcType,
                                name: ifcName,
                            })
                        } else {
                            setBimTooltip(null)
                        }
                    } else {
                        setBimTooltip(null)
                    }
                } else {
                    setBimTooltip(null)
                }
            }
        }
        container.addEventListener('mousemove', handleMouseMove)

        // Track right-click down position for drag detection
        let rightDownPos = { x: 0, y: 0 }
        const handleRightDown = (e: MouseEvent) => {
            if (e.button === 2) rightDownPos = { x: e.clientX, y: e.clientY }
        }
        container.addEventListener('mousedown', handleRightDown)

        // Right-click handler: BIM context menu (only if not dragging)
        const handleContextMenu = (e: MouseEvent) => {
            // Suppress if user was orbiting (dragged > 5px)
            const dx = e.clientX - rightDownPos.x
            const dy = e.clientY - rightDownPos.y
            if (Math.sqrt(dx * dx + dy * dy) > 5) return

            const bimGroup = bimGroupRef.current
            const camera = cameraRef.current
            if (!bimGroup || !camera || bimGroup.children.length === 0 || !bimGroup.visible) return
            const rect = container.getBoundingClientRect()
            const mouse = new THREE.Vector2(
                ((e.clientX - rect.left) / rect.width) * 2 - 1,
                -((e.clientY - rect.top) / rect.height) * 2 + 1
            )
            const rc = camRaycasterRef.current
            rc.setFromCamera(mouse, camera)
            const bimHits = rc.intersectObjects(bimGroup.children, true)
              .filter(h => h.object.visible)
            if (bimHits.length > 0) {
                e.preventDefault()
                e.stopPropagation()
                const hit = bimHits[0].object as THREE.Mesh
                const mat = hit.material as THREE.MeshStandardMaterial
                setBimCtxMenu({
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top,
                    expressID: hit.userData?.expressID ?? -1,
                    type: hit.userData?.ifc_type || '',
                    name: hit.userData?.ifc_name || '',
                    opacity: mat.opacity ?? 1.0,
                })
                setBimTooltip(null)
            }
        }
        container.addEventListener('contextmenu', handleContextMenu)

        // Click handler: fly to camera pose
        const handleClick = (e: MouseEvent) => {
            const camGroup = cameraGroupRef.current
            const camera = cameraRef.current
            const controls = controlsRef.current
            if (!camGroup || !camera || !controls || camGroup.children.length === 0 || !showCameraPosesRef.current) return
            const rect = container.getBoundingClientRect()
            const mouse = new THREE.Vector2(
                ((e.clientX - rect.left) / rect.width) * 2 - 1,
                -((e.clientY - rect.top) / rect.height) * 2 + 1
            )
            const rc = camRaycasterRef.current
            rc.setFromCamera(mouse, camera)
            const spheres = camGroup.children.filter(c => (c as THREE.Mesh).isMesh && c.userData.isCamPose)
            const hits = rc.intersectObjects(spheres, false)
            if (hits.length > 0) {
                const hit = hits[0].object
                const pos = hit.position.clone()
                const lookDir = hit.userData.lookDir as THREE.Vector3
                if (lookDir) {
                    // Move camera to pose position, look along pose direction
                    camera.position.copy(pos)
                    controls.target.copy(pos).addScaledVector(lookDir, 0.5)
                    // Set FOV from intrinsics if available
                    const intr = hit.userData.intrinsics
                    if (intr && intr.fy && intr.cy) {
                        const fovRad = 2 * Math.atan(intr.cy / intr.fy)
                        camera.fov = fovRad * 180 / Math.PI
                        camera.updateProjectionMatrix()
                    }
                    controls.update()
                }
            }
        }
        container.addEventListener('click', handleClick)

        // Close BIM context menu on any left click
        const handleLeftClick = () => setBimCtxMenu(null)
        container.addEventListener('click', handleLeftClick)

        return () => {
            container.removeEventListener('mousemove', handleMouseMove)
            container.removeEventListener('mousedown', handleRightDown)
            container.removeEventListener('contextmenu', handleContextMenu)
            container.removeEventListener('click', handleClick)
            container.removeEventListener('click', handleLeftClick)
        }
    }, [])

    // When activeSession changes, send load command on existing WS
    // Skip if pipeline is running — no cloud exists yet
    useEffect(() => {
        if (!activeSession) return
        if (pipelineRunningRef.current) return  // Pipeline in progress, cloud doesn't exist yet
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
            // Reset group transform (gizmo may have left a rotation on it)
            group.matrix.identity()
            group.position.set(0, 0, 0)
            group.quaternion.identity()
            group.scale.set(1, 1, 1)
            group.matrixAutoUpdate = true
            group.matrixWorldNeedsUpdate = true
        }
        // Clear BIM models
        const bimGroup = bimGroupRef.current
        if (bimGroup) {
            while (bimGroup.children.length > 0) {
                const child = bimGroup.children[0]
                bimGroup.remove(child)
                child.traverse((obj: any) => {
                    if (obj.geometry) obj.geometry.dispose()
                    if (obj.material) {
                        if (Array.isArray(obj.material)) obj.material.forEach((m: any) => m.dispose())
                        else obj.material.dispose()
                    }
                })
            }
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

        // Auto-center camera on first chunk (only if session not already framed and not restoring)
        if (totalPointsRef.current === newPointCount && controlsRef.current && cameraRef.current && !sessionFramedRef.current && !preserveCameraRef.current) {
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
            const serverHasConfidence = msg.hasConfidence as boolean | undefined
            // console.log(`[Viewport] Potree ready: ${pts?.toLocaleString()} points at ${url}`)
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
                // Guard: if this loader was replaced (e.g. by sábana), skip
                if (potreeLoaderRef.current !== loader) return
                if (onStatusMessage) onStatusMessage(`LOD octree loaded — ${pts?.toLocaleString()} total points`)
                onPointCount(loadedPts)
                if (onHasConfidence) onHasConfidence(!!serverHasConfidence)

                // Apply floor alignment transform if provided
                if (floorTransform && floorTransform.length === 16) {
                    loader.setTransform(floorTransform)
                    // Store for camera centering only — OBBs are already
                    // floor-aligned on the backend so they don't need this
                    floorTransformRef.current = new THREE.Matrix4().fromArray(floorTransform)
                }

                // Auto-center camera on cloud bounding box (world-space)
                const bbox = loader.getWorldBoundingBox()
                if (bbox && cameraRef.current && controlsRef.current) {
                    // Store for resetCamera
                    cloudBBoxRef.current = bbox.clone()
                    // If BIM already loaded (race: BIM resolved before cloud),
                    // expand bbox to include BIM geometry
                    const bimGroup = bimGroupRef.current
                    if (bimGroup && bimGroup.children.length > 0) {
                        const bimBox = new THREE.Box3().setFromObject(bimGroup)
                        if (!bimBox.isEmpty()) {
                            cloudBBoxRef.current.union(bimBox)
                        }
                    }
                    // Frame with combined bbox — only on first load for this session
                    const sid = msg.session_id as string
                    if (sessionFramedRef.current !== sid && !preserveCameraRef.current) {
                        frameCloud(cloudBBoxRef.current, cameraRef.current, controlsRef.current)
                    }
                    sessionFramedRef.current = sid
                    // Restore saved camera state if preserving
                    if (preserveCameraRef.current && cameraRef.current && controlsRef.current) {
                        cameraRef.current.position.copy(preserveCameraRef.current.pos)
                        controlsRef.current.target.copy(preserveCameraRef.current.target)
                        controlsRef.current.update()
                    }
                    // Adapt grid to combined extents
                    if (sceneRef.current) adaptGrid(cloudBBoxRef.current, sceneRef.current, gridRef, showGridRef.current)
                }

                // Render camera poses as arrows if provided
                const cameraPoses = msg.cameraPoses as number[][][] | undefined
                if (cameraPoses && cameraGroupRef.current) {
                    const camGroup = cameraGroupRef.current
                    // Clear previous
                    while (camGroup.children.length > 0) camGroup.remove(camGroup.children[0])
                    const frameNames = msg.cameraFrameNames as string[] | undefined
                    const intrinsics = msg.cameraIntrinsics as Array<{ fx: number, fy: number, cx: number, cy: number }> | undefined
                    const arrowLen = 0.3
                    cameraPoses.forEach((c2w, idx) => {
                        const pos = new THREE.Vector3(c2w[0][3], c2w[1][3], c2w[2][3])
                        // OpenCV convention: camera looks along +Z
                        const lookDir = new THREE.Vector3(c2w[0][2], c2w[1][2], c2w[2][2]).normalize()

                        const hue = idx / cameraPoses.length
                        const color = new THREE.Color().setHSL(hue, 0.9, 0.55)

                        // Small sphere at camera position
                        const sphere = new THREE.Mesh(
                            new THREE.SphereGeometry(0.03, 6, 4),
                            new THREE.MeshBasicMaterial({ color })
                        )
                        sphere.position.copy(pos)
                        sphere.userData = {
                            isCamPose: true,
                            frameName: frameNames?.[idx] || `frame_${idx}`,
                            sessionId: msg.session_id,
                            lookDir: lookDir.clone(),
                            intrinsics: intrinsics?.[idx] || null,
                        }
                        camGroup.add(sphere)

                        // Arrow for look direction
                        const lookArrow = new THREE.ArrowHelper(lookDir, pos, arrowLen, color.getHex(), 0.06, 0.04)
                        camGroup.add(lookArrow)
                    })
                    console.log(`[Viewport] Rendered ${cameraPoses.length} camera poses`)
                    if (onHasCameraPoses) onHasCameraPoses(true)
                }
            }).catch((err) => {
                console.error('[Viewport] Potree load failed:', err)
                if (onStatusMessage) onStatusMessage(`Potree load error: ${err.message}`)
            })
            return // Don't process further
        }

        // v2.0 format: { instances: [{label, instance_id, color, total_points, obb, ...}] }
        if (Array.isArray(msg.instances)) {
            // Reset obbGroup transform — new OBBs are computed in the current
            // display space (from floor_transform.npz). If the gizmo was saved,
            // the obbGroup may still have the old gizmo delta applied. Clear it
            // so the fresh OBBs render at their correct positions.
            const obbGroup = obbGroupRef.current
            if (obbGroup) {
                obbGroup.matrix.identity()
                obbGroup.matrixAutoUpdate = true
                obbGroup.matrixWorldNeedsUpdate = true
                obbGroup.position.set(0, 0, 0)
                obbGroup.quaternion.identity()
                obbGroup.scale.set(1, 1, 1)
            }
            renderOBBs(msg.instances as Array<Record<string, unknown>>)
        }
        if (msg.type === 'status' || msg.type === 'progress') {
            const text = (msg.message || msg.status || '') as string
            if (text && onStatusMessage) onStatusMessage(text)
        }
        if (msg.type === 'pipeline_progress' && onPipelineProgress) {
            onPipelineProgress(msg)
        }
        // ── Sábana: load via Potree, dim BIM, hide OBBs ──
        if (msg.type === 'sabana_potree_ready') {
            const url = msg.url as string
            const nPts = msg.points as number

            // Bump load counter — any in-flight loads with an older ID are stale
            const thisLoadId = ++sabanaLoadIdRef.current
            console.log(`[Viewport] Sábana Potree ready (load #${thisLoadId}): ${nPts?.toLocaleString()} points at ${url}`)
            if (onStatusMessage) onStatusMessage(`Loading sábana LOD octree (${nPts?.toLocaleString()} points)...`)

            const scene = sceneRef.current
            const camera = cameraRef.current
            const mat = materialRef.current
            if (!scene || !camera || !mat) return

            // 1) Clear only the scan cloud (not BIM, not OBBs)
            if (potreeLoaderRef.current) {
                potreeLoaderRef.current.dispose()
                potreeLoaderRef.current = null
            }
            // Clear legacy geometry (non-Potree binary streaming)
            const geo = geometryRef.current
            if (geo) {
                geo.deleteAttribute('position')
                geo.deleteAttribute('color')
                geo.deleteAttribute('classId')
            }
            totalPointsRef.current = 0

            // 2) Load sábana via PotreeOctreeLoader (forceClassId=-1 → always visible)
            const loader = new PotreeOctreeLoader(scene, camera, mat, undefined, -1)
            potreeLoaderRef.current = loader
            loader.load(url).then((loadedPts) => {
                // Guard: if a newer load was started, discard this stale result
                if (sabanaLoadIdRef.current !== thisLoadId) {
                    console.log(`[Viewport] Discarding stale sábana load #${thisLoadId} (current: #${sabanaLoadIdRef.current})`)
                    return
                }
                console.log(`[Viewport] Sábana Potree loaded (load #${thisLoadId}): ${loadedPts.toLocaleString()} points`)
                if (onStatusMessage) onStatusMessage(`Sábana: ${nPts?.toLocaleString()} deviation points`)
                onPointCount(loadedPts)

                // Keep camera where it is — user navigates from current position

                // 3) Dim BIM meshes to 20% opacity
                const bimGroup = bimGroupRef.current
                if (bimGroup) {
                    let dimmed = 0
                    bimGroup.traverse((child) => {
                        if (!(child instanceof THREE.Mesh)) return
                        if (!child.userData._originalMaterial) {
                            child.userData._originalMaterial = child.material
                        }
                        child.material = new THREE.MeshBasicMaterial({
                            color: 0x888888,
                            transparent: true,
                            opacity: 0.20,
                            side: THREE.DoubleSide,
                            depthWrite: false,
                        })
                        dimmed++
                    })
                    console.log(`[Viewport] Dimmed ${dimmed} BIM meshes`)
                } else {
                    console.warn('[Viewport] No bimGroup to dim!')
                }

                // 4) Hide OBB wireframes (segmentation bboxes are noise in sábana mode)
                const obbGroup = obbGroupRef.current
                if (obbGroup) obbGroup.visible = false

                if (onSabanaLoaded) onSabanaLoaded(loadedPts)
            }).catch((err) => {
                if (sabanaLoadIdRef.current !== thisLoadId) return  // stale, ignore error too
                console.error('[Viewport] Sábana Potree load error:', err)
                if (onStatusMessage) onStatusMessage(`Sábana load error: ${err.message}`)
            })
        }
        // ── BIM: load IFC models via web-ifc ──
        if (msg.type === 'bim_ready') {
            const models = msg.models as Array<{ name: string; url: string }>
            if (!models || models.length === 0) return
            const bimGroup = bimGroupRef.current
            if (!bimGroup) return

            import('./IFCLoader').then(({ loadIFC }) => {
                const loadedResults: import('./IFCLoader').IFCLoadResult[] = []
                let remaining = models.length
                for (const model of models) {
                    // console.log(`[Viewport] Loading BIM: ${model.name}`)
                    if (onStatusMessage) onStatusMessage(`Loading BIM: ${model.name}...`)
                    loadIFC(model.url, model.name).then((result) => {
                        bimGroup.add(result.group)
                        const bimBox = new THREE.Box3().setFromObject(result.group)
                        // Expand scene bbox to include BIM, re-frame + re-grid
                        if (!bimBox.isEmpty()) {
                            const bimSize = new THREE.Vector3()
                            bimBox.getSize(bimSize)
                            console.log(`[Viewport] BIM bbox: min(${bimBox.min.x.toFixed(2)},${bimBox.min.y.toFixed(2)},${bimBox.min.z.toFixed(2)}) max(${bimBox.max.x.toFixed(2)},${bimBox.max.y.toFixed(2)},${bimBox.max.z.toFixed(2)}) size(${bimSize.x.toFixed(2)},${bimSize.y.toFixed(2)},${bimSize.z.toFixed(2)})`)
                            if (cloudBBoxRef.current) {
                                const cloudSize = new THREE.Vector3()
                                cloudBBoxRef.current.getSize(cloudSize)
                                console.log(`[Viewport] Cloud bbox before union: min(${cloudBBoxRef.current.min.x.toFixed(2)},${cloudBBoxRef.current.min.y.toFixed(2)},${cloudBBoxRef.current.min.z.toFixed(2)}) max(${cloudBBoxRef.current.max.x.toFixed(2)},${cloudBBoxRef.current.max.y.toFixed(2)},${cloudBBoxRef.current.max.z.toFixed(2)}) size(${cloudSize.x.toFixed(2)},${cloudSize.y.toFixed(2)},${cloudSize.z.toFixed(2)})`)
                                cloudBBoxRef.current.union(bimBox)
                                const unionSize = new THREE.Vector3()
                                cloudBBoxRef.current.getSize(unionSize)
                                console.log(`[Viewport] Union bbox: min(${cloudBBoxRef.current.min.x.toFixed(2)},${cloudBBoxRef.current.min.y.toFixed(2)},${cloudBBoxRef.current.min.z.toFixed(2)}) max(${cloudBBoxRef.current.max.x.toFixed(2)},${cloudBBoxRef.current.max.y.toFixed(2)},${cloudBBoxRef.current.max.z.toFixed(2)}) size(${unionSize.x.toFixed(2)},${unionSize.y.toFixed(2)},${unionSize.z.toFixed(2)}) maxDim=${Math.max(unionSize.x, unionSize.y, unionSize.z).toFixed(2)}`)
                            } else {
                                cloudBBoxRef.current = bimBox.clone()
                            }
                            if (cameraRef.current && controlsRef.current) {
                                if (!preserveCameraRef.current) {
                                    frameCloud(cloudBBoxRef.current, cameraRef.current, controlsRef.current)
                                }
                            }
                            // Restore saved camera state if mode-switching (final load step)
                            if (preserveCameraRef.current && cameraRef.current && controlsRef.current) {
                                cameraRef.current.position.copy(preserveCameraRef.current.pos)
                                controlsRef.current.target.copy(preserveCameraRef.current.target)
                                controlsRef.current.update()
                                preserveCameraRef.current = null
                            }
                            if (sceneRef.current) {
                                adaptGrid(cloudBBoxRef.current, sceneRef.current, gridRef, showGridRef.current)
                            }
                        }
                        console.log(`[Viewport] ✅ BIM loaded: ${model.name} (${result.group.children.length} elements, ${result.hierarchy.length} hierarchy roots)`)
                        if (onStatusMessage) onStatusMessage(`BIM loaded: ${model.name} (${result.group.children.length} elements)`)
                        loadedResults.push(result)
                        remaining--
                        if (remaining === 0 && onBimLoaded) {
                            onBimLoaded(loadedResults)
                        }
                    }).catch((err) => {
                        console.error(`[Viewport] BIM load error: ${model.name}`, err)
                        if (onStatusMessage) onStatusMessage(`BIM error: ${err.message}`)
                        remaining--
                        if (remaining === 0 && onBimLoaded && loadedResults.length > 0) {
                            onBimLoaded(loadedResults)
                        }
                    })
                }
            })
        }
    }, [onStatusMessage, onSegments, onPipelineProgress, onPointCount, onBimLoaded, onSabanaLoaded])

    // Toggle camera poses visibility
    useEffect(() => {
        const g = cameraGroupRef.current
        if (g) g.visible = showCameraPoses
        showCameraPosesRef.current = showCameraPoses
        // Clear any lingering tooltips when hiding
        if (!showCameraPoses) {
            setCamTooltip(null)
            setBimTooltip(null)
        }
    }, [showCameraPoses])

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
                id: instId,
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

            // Create a group for wireframe + label
            const obbContainer = new THREE.Group()
            obbContainer.position.set(center[0], center[1], center[2])

            // Apply rotation from 3x3 matrix
            if (rotation && rotation.length === 3) {
                const rotMatrix = new THREE.Matrix4()
                rotMatrix.set(
                    rotation[0][0], rotation[0][1], rotation[0][2], 0,
                    rotation[1][0], rotation[1][1], rotation[1][2], 0,
                    rotation[2][0], rotation[2][1], rotation[2][2], 0,
                    0, 0, 0, 1
                )
                obbContainer.setRotationFromMatrix(rotMatrix)
            }

            obbContainer.add(wireframe)

            // ── Voxel mesh: semi-transparent surface quads ──
            const voxelMesh = inst.voxel_mesh as { voxel_size: number; count: number; data: number[][] } | undefined
            if (voxelMesh && voxelMesh.count > 0) {
                const vs = voxelMesh.voxel_size
                const planeGeo = new THREE.PlaneGeometry(vs, vs)
                const planeMat = new THREE.MeshBasicMaterial({
                    color: hexColor,
                    opacity: 0.25,
                    transparent: true,
                    side: THREE.DoubleSide,
                    depthWrite: false,
                })
                const mesh = new THREE.InstancedMesh(planeGeo, planeMat, voxelMesh.count)

                const dummy = new THREE.Object3D()
                const defaultNormal = new THREE.Vector3(0, 0, 1)
                const quat = new THREE.Quaternion()

                for (let vi = 0; vi < voxelMesh.count; vi++) {
                    const d = voxelMesh.data[vi]
                    dummy.position.set(d[0], d[1], d[2])
                    const voxNormal = new THREE.Vector3(d[3], d[4], d[5]).normalize()
                    quat.setFromUnitVectors(defaultNormal, voxNormal)
                    dummy.quaternion.copy(quat)
                    dummy.updateMatrix()
                    mesh.setMatrixAt(vi, dummy.matrix)
                }
                mesh.instanceMatrix.needsUpdate = true
                group.add(mesh)

                // ── Wireframe: connect neighboring voxel centers via grid hash (O(n)) ──
                const linePositions: number[] = []

                // Build spatial hash: grid key → voxel index
                const gridMap = new Map<string, number>()
                for (let vi = 0; vi < voxelMesh.count; vi++) {
                    const d = voxelMesh.data[vi]
                    const gx = Math.floor(d[0] / vs)
                    const gy = Math.floor(d[1] / vs)
                    const gz = Math.floor(d[2] / vs)
                    gridMap.set(`${gx},${gy},${gz}`, vi)
                }

                // For each voxel, check 13 unique neighbor directions (avoid duplicates)
                const neighborOffsets = [
                    [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1],
                    [0, 1, 1], [0, 1, -1],
                    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                ]
                for (let vi = 0; vi < voxelMesh.count; vi++) {
                    const d = voxelMesh.data[vi]
                    const gx = Math.floor(d[0] / vs)
                    const gy = Math.floor(d[1] / vs)
                    const gz = Math.floor(d[2] / vs)
                    for (const [ox, oy, oz] of neighborOffsets) {
                        const key = `${gx + ox},${gy + oy},${gz + oz}`
                        const ni = gridMap.get(key)
                        if (ni !== undefined) {
                            const nd = voxelMesh.data[ni]
                            linePositions.push(d[0], d[1], d[2], nd[0], nd[1], nd[2])
                        }
                    }
                }

                if (linePositions.length > 0) {
                    const lineGeo = new THREE.BufferGeometry()
                    lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3))
                    const lineMat = new THREE.LineBasicMaterial({
                        color: hexColor,
                        opacity: 0.7,
                        transparent: true,
                        depthWrite: false,
                    })
                    const wireLines = new THREE.LineSegments(lineGeo, lineMat)
                    group.add(wireLines)
                }
            }

            // Text label sprite at top of box
            const labelCanvas = document.createElement('canvas')
            const ctx = labelCanvas.getContext('2d')!
            ctx.font = 'bold 28px sans-serif'
            const textWidth = ctx.measureText(label).width
            labelCanvas.width = Math.max(textWidth + 20, 64)
            labelCanvas.height = 36
            ctx.font = 'bold 28px sans-serif'
            ctx.fillStyle = 'rgba(0,0,0,0.6)'
            ctx.roundRect(0, 0, labelCanvas.width, labelCanvas.height, 6)
            ctx.fill()
            ctx.fillStyle = colorStr
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            ctx.fillText(label, labelCanvas.width / 2, labelCanvas.height / 2)
            const labelTex = new THREE.CanvasTexture(labelCanvas)
            const labelMat = new THREE.SpriteMaterial({ map: labelTex, depthTest: false })
            const labelSprite = new THREE.Sprite(labelMat)
            const aspect = labelCanvas.width / labelCanvas.height
            const spriteH = 0.08
            labelSprite.scale.set(spriteH * aspect, spriteH, 1)
            labelSprite.position.set(0, halfExtents[1] + spriteH * 0.7, 0)
            obbContainer.add(labelSprite)

            group.add(obbContainer)
            obbMapRef.current.set(globalKey, obbContainer)
            geom.dispose()
        }

        // NOTE: OBBs are already in floor-aligned coordinates from the backend
        // (segmentation_pipeline.py applies floor transform to xyz_display before
        // computing OBBs). No additional transform needed here.


        // console.log(`[Viewport] Rendered ${group.children.length} OBBs`)

        // Notify parent with segment list
        if (onSegments) onSegments(segmentList)
    }, [onSegments])

    return (
        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
            <div ref={containerRef} className="viewport-canvas" style={{ width: '100%', height: '100%' }} />
            {activeTool === 'align' && (
                <div style={{
                    position: 'absolute', top: 12, right: 12, background: 'rgba(20,20,30,0.92)',
                    borderRadius: 10, padding: '14px 18px', color: '#fff', fontSize: 13,
                    display: 'flex', flexDirection: 'column', gap: 10, minWidth: 180,
                    border: '1px solid rgba(255,255,255,0.12)', backdropFilter: 'blur(8px)',
                    zIndex: 100
                }}>
                    <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 2 }}>⛶ Align Cloud</div>
                    <div style={{ display: 'flex', gap: 6 }}>
                        <button
                            style={{ flex: 1, padding: '6px 0', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 12, fontWeight: 600, background: alignMode === 'rotate' ? '#f39c12' : '#444', color: '#fff' }}
                            onClick={() => setAlignMode('rotate')}
                        >🔄 Rotate</button>
                        <button
                            style={{ flex: 1, padding: '6px 0', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 12, fontWeight: 600, background: alignMode === 'translate' ? '#3498db' : '#444', color: '#fff' }}
                            onClick={() => setAlignMode('translate')}
                        >↔️ Move</button>
                    </div>
                    <hr style={{ border: 'none', borderTop: '1px solid rgba(255,255,255,0.1)', margin: '2px 0' }} />
                    <button
                        style={{ padding: '8px 0', border: 'none', borderRadius: 6, cursor: alignDirty ? 'pointer' : 'not-allowed', fontSize: 13, fontWeight: 700, background: alignDirty ? '#2ecc71' : '#333', color: '#fff', opacity: alignDirty ? 1 : 0.5 }}
                        onClick={saveAlignment}
                        disabled={!alignDirty}
                    >💾 Save Alignment</button>
                </div>
            )}
            {/* Camera pose hover tooltip */}
            {camTooltip && (
                <div style={{
                    position: 'absolute',
                    left: camTooltip.x + 16,
                    top: camTooltip.y - 80,
                    background: 'rgba(10,10,20,0.95)',
                    borderRadius: 8,
                    padding: 8,
                    color: '#fff',
                    fontSize: 11,
                    fontWeight: 600,
                    zIndex: 200,
                    pointerEvents: 'none',
                    border: '1px solid rgba(255,255,255,0.2)',
                    backdropFilter: 'blur(6px)',
                    minWidth: 120,
                    maxWidth: 220,
                }}>
                    <div style={{ marginBottom: 4 }}>{camTooltip.frameName}</div>
                    <img
                        src={`/api/sessions/${camTooltip.sessionId}/frames/${camTooltip.frameName}`}
                        alt={camTooltip.frameName}
                        style={{ width: '100%', borderRadius: 4, display: 'block' }}
                        onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                    />
                </div>
            )}
            {/* BIM element hover tooltip */}
            {bimTooltip && !camTooltip && (
                <div style={{
                    position: 'absolute',
                    left: bimTooltip.x + 14,
                    top: bimTooltip.y + 14,
                    background: 'rgba(10,10,20,0.92)',
                    borderRadius: 6,
                    padding: '5px 10px',
                    color: '#fff',
                    fontSize: 11,
                    zIndex: 200,
                    pointerEvents: 'none',
                    border: '1px solid rgba(100,180,255,0.3)',
                    backdropFilter: 'blur(6px)',
                    maxWidth: 360,
                    wordBreak: 'break-all' as const,
                }}>
                    <span style={{ color: '#7cb8ff', fontWeight: 600 }}>{bimTooltip.type.replace('Ifc', '')}</span>
                    {bimTooltip.name && !bimTooltip.name.startsWith('Element_') && (
                        <span style={{ marginLeft: 6, opacity: 0.8 }}>{bimTooltip.name}</span>
                    )}
                </div>
            )}
            {/* BIM right-click context menu */}
            {bimCtxMenu && (
                <div style={{
                    position: 'absolute',
                    left: bimCtxMenu.x,
                    top: bimCtxMenu.y,
                    background: 'rgba(20,20,30,0.97)',
                    borderRadius: 8,
                    padding: 0,
                    color: '#fff',
                    fontSize: 12,
                    zIndex: 300,
                    border: '1px solid rgba(100,180,255,0.25)',
                    backdropFilter: 'blur(10px)',
                    minWidth: 200,
                    boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
                    overflow: 'hidden',
                }} onClick={e => e.stopPropagation()}>
                    {/* Header */}
                    <div style={{
                        padding: '8px 12px',
                        borderBottom: '1px solid rgba(255,255,255,0.1)',
                        fontSize: 11,
                        opacity: 0.7,
                        wordBreak: 'break-all' as const,
                    }}>
                        <span style={{ color: '#7cb8ff', fontWeight: 600 }}>{bimCtxMenu.type.replace('Ifc', '')}</span>
                        {bimCtxMenu.name && !bimCtxMenu.name.startsWith('Element_') && (
                            <span style={{ marginLeft: 6 }}>{bimCtxMenu.name}</span>
                        )}
                    </div>
                    {/* Transparency slider */}
                    <div style={{ padding: '8px 12px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontSize: 11, opacity: 0.7, minWidth: 70 }}>Transparency</span>
                            <input
                                type="range" min="0" max="100"
                                value={Math.round((1 - bimCtxMenu.opacity) * 100)}
                                onChange={e => {
                                    const newOpacity = 1 - parseInt(e.target.value) / 100
                                    const bimGroup = bimGroupRef.current
                                    if (!bimGroup) return
                                    bimGroup.traverse(child => {
                                        if ((child as THREE.Mesh).isMesh && child.userData?.expressID === bimCtxMenu.expressID) {
                                            const mat = (child as THREE.Mesh).material as THREE.MeshStandardMaterial
                                            // Save original state on first modification
                                            if (child.userData._origOpacity === undefined) {
                                                child.userData._origOpacity = mat.opacity
                                                child.userData._origTransparent = mat.transparent
                                                child.userData._origDepthWrite = mat.depthWrite
                                            }
                                            mat.opacity = newOpacity
                                            mat.transparent = true
                                            mat.depthWrite = newOpacity > 0.9
                                            mat.needsUpdate = true
                                        }
                                    })
                                    setBimCtxMenu(prev => prev ? { ...prev, opacity: newOpacity } : null)
                                }}
                                style={{ flex: 1, accentColor: '#7cb8ff' }}
                            />
                            <span style={{ fontSize: 10, opacity: 0.5, minWidth: 28, textAlign: 'right' }}>
                                {Math.round((1 - bimCtxMenu.opacity) * 100)}%
                            </span>
                        </div>
                    </div>
                    {/* Hide */}
                    <div
                        style={{ padding: '8px 12px', cursor: 'pointer', borderBottom: '1px solid rgba(255,255,255,0.06)' }}
                        onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.08)')}
                        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                        onClick={() => {
                            const bimGroup = bimGroupRef.current
                            if (!bimGroup) return
                            bimGroup.traverse(child => {
                                if ((child as THREE.Mesh).isMesh && child.userData?.expressID === bimCtxMenu.expressID) {
                                    child.visible = false
                                }
                            })
                            setBimCtxMenu(null)
                        }}
                    >👁‍🗨 Hide</div>
                    {/* Isolate */}
                    <div
                        style={{ padding: '8px 12px', cursor: 'pointer', borderBottom: '1px solid rgba(255,255,255,0.06)' }}
                        onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.08)')}
                        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                        onClick={() => {
                            const bimGroup = bimGroupRef.current
                            if (!bimGroup) return
                            bimGroup.traverse(child => {
                                if ((child as THREE.Mesh).isMesh && child.userData?.expressID !== undefined) {
                                    child.visible = child.userData.expressID === bimCtxMenu.expressID
                                }
                            })
                            setBimCtxMenu(null)
                        }}
                    >🔍 Isolate</div>
                    {/* Show all */}
                    <div
                        style={{ padding: '8px 12px', cursor: 'pointer' }}
                        onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.08)')}
                        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                        onClick={() => {
                            const bimGroup = bimGroupRef.current
                            if (!bimGroup) return
                            bimGroup.traverse(child => {
                                if ((child as THREE.Mesh).isMesh) {
                                    child.visible = true
                                    const mat = (child as THREE.Mesh).material as THREE.MeshStandardMaterial
                                    // Restore original material state (preserves IFC glass transparency)
                                    const origOpacity = child.userData._origOpacity ?? mat.opacity
                                    const origTransparent = child.userData._origTransparent ?? mat.transparent
                                    const origDepthWrite = child.userData._origDepthWrite ?? mat.depthWrite
                                    mat.opacity = origOpacity
                                    mat.transparent = origTransparent
                                    mat.depthWrite = origDepthWrite
                                    mat.needsUpdate = true
                                    // Clear saved state
                                    delete child.userData._origOpacity
                                    delete child.userData._origTransparent
                                    delete child.userData._origDepthWrite
                                }
                            })
                            setBimCtxMenu(null)
                        }}
                    >✨ Show All</div>
                </div>
            )}
        </div>
    )
})

export default Viewport
