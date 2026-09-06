/**
 * STAC Build — 3D Viewport Component
 * Three.js-based point cloud renderer
 * Hernán Barreto — Ingerop IN3
 */
import { useRef, useEffect, useCallback, forwardRef, useImperativeHandle, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { PotreeOctreeLoader } from './PotreeLoader'
import { AssistantViz, type SceneObject, type UserVolume, type TraceEntry } from './assistantViz'

// Factory for the shared GLTFLoader. The TSDF/recon .glb files are compressed
// with EXT_meshopt_compression (geometry) + WebP textures (see tools/glb/), so
// the loader MUST have a Meshopt decoder wired or those meshes silently fail to
// parse. Uncompressed GLBs ignore the decoder, so this stays backwards-compatible.
function makeGltfLoader(): GLTFLoader {
    const l = new GLTFLoader()
    l.setMeshoptDecoder(MeshoptDecoder)
    return l
}

type Tool = 'navigate' | 'measure-distance' | 'measure-angle' | 'section-box' | 'align' | 'erase'

interface ViewportProps {
    pointSize: number
    pointBudget: number
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
    /** Gizmo edit finished on an evaluation volume — persist + re-evaluate. */
    onVolumeChanged?: (params: { volume_id: number; center: number[]; size: number[]; yaw_deg: number }) => void
    eraseRadius?: number
    eraseShape?: 'sphere' | 'cube' | 'box'
    eraseYawDeg?: number
    onEraseRadiusChange?: (r: number) => void
    onEraseMarksChanged?: (n: number) => void
    onEraseBoxSelected?: (selected: boolean) => void
    /** A placed library object was selected/deselected (id | null). */
    onSceneObjectSelected?: (id: number | null) => void
    /** The placed-objects list changed (load/add/remove/visibility) — App
        mirrors it in the panel's Objects section. */
    onSceneObjectsChanged?: (objs: Array<{ id: number; name: string; visible: boolean }>) => void
    /** Brush transaction committed — App refreshes panel counters (ledger
        may be null when the backend predates the trace protocol). */
    onEraseLedger?: (ledger: unknown) => void
    /** Freshly-published meshes (poisson/pgsr stage done) — App refreshes
        the mesh panel list. */
    onTsdfReady?: (sessionId: string, stage: string) => void
    /** Volume deleted from the viewer toolbar. */
    onVolumeDeleted?: (volumeId: number) => void
    /** Chunk box selected (null = deselected) — App shows the Save panel. */
    onChunkSelected?: (chunk: number | null) => void
    /** Chunk box dragged: delta transform in the cloud FILE frame
        (row-major 4x4). null = box back at its origin. */
    onChunkDelta?: (chunk: number, matrixRowMajor: number[] | null) => void
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
    commitErase: (target?: number | null, newLabel?: string, onlyInstances?: number[], includeUnsegmented?: boolean) => Promise<void>
    /** Placed library objects (references only — sources never touched). */
    placeSceneObject: (entry: { id: number; name: string; url: string; matrix?: number[] }) => Promise<void>
    reloadSceneObjects: (sessionId: string) => Promise<void>
    setSceneObjectMode: (m: 'translate' | 'rotate' | 'scale') => void
    removeSelectedSceneObject: () => Promise<void>
    removeSceneObject: (id: number) => Promise<void>
    setSceneObjectVisible: (id: number, visible: boolean) => void
    getSceneAlignTargets: () => Array<{ key: string; label: string }>
    alignSceneObject: (op: 'floor' | 'same_base' | 'on_top' | 'center_xz' | 'center_y', targetKey?: string) => void
    /** Preview the brush confidence filter: points below thr light up red. null = off. */
    setConfHighlight: (thr: number | null) => void
    /** Apply the brush confidence filter: visible segments' points below thr → unsegmented. */
    applyConfidenceFilter: (thr: number, onlyInstances?: number[], includeUnsegmented?: boolean) => Promise<void>
    setEraseBoxMode: (m: 'translate' | 'rotate' | 'scale') => void
    removeSelectedEraseBox: () => void
    clearEraseMarks: () => void
    resetSectionBox: () => void
    resetCamera: () => void
    clearScene: () => void
    refreshSegmentOBBs: (sessionId: string) => void
    setFloorTransform: (arr: number[]) => void
    setOBBsVisible: (visible: boolean) => void
    setSegmentVisibility: (segId: number, visible: boolean) => void
    setCloudObjectVisible: (visible: boolean) => void
    reloadShapes: (sessionId: string) => Promise<void>
    setShapeVisibility: (instanceId: number, visible: boolean) => void
    clearShapes: () => void
    reloadTsdf: (sessionId: string) => Promise<void>
    setTsdfVisibility: (folder: string, visible: boolean) => void
    clearTsdf: () => void
    reloadReconScene: (sessionId: string) => Promise<void>
    setReconElementVisibility: (instanceId: number, visible: boolean) => void
    clearReconScene: () => void
    // Flythrough (synced video↔3D): drive the camera to a per-frame c2w pose.
    setFlythroughActive: (active: boolean) => void
    setCameraToPose: (c2wRowMajor: number[]) => void
    // Match the 3D camera's vertical FOV to the real camera (from intrinsics) so
    // the flythrough frames the scene exactly like the video. Restored on close.
    setCameraFov: (fovYDeg: number) => void
    // Immersive assistant: feed scene objects (for OBB lookup), animate the
    // geometry behind a spatial answer, and manage user-defined volumes.
    setAssistantObjects: (objects: SceneObject[]) => void
    visualizeMeasurement: (trace: TraceEntry[]) => void
    clearAssistantViz: () => void
    addUserVolume: (volume: UserVolume) => void
    removeUserVolume: (volumeId: number) => void
    setVolumeStatus: (volumeId: number, status: 'free' | 'touching' | 'colliding') => void
    setVolumeSolid: (volumeId: number, solid: boolean) => void
    frameBox: (min: number[], max: number[]) => void
    // Chunk boxes (USER 2026-09-06): floor-aligned OBB per chunk, all
    // deselected by default; selected box gets a translate/rotate gizmo
    // centered on the chunk. Delta is reported in the CLOUD FILE frame
    // (row-major 4x4) for /api/segmentation/chunks/apply_transform.
    setChunkBoxes: (boxes: Array<{ chunk: number; center: number[]; size: number[]; yaw: number }>) => void
    setChunkGizmoMode: (m: 'translate' | 'rotate') => void
    resetChunkBox: (chunk: number) => void
    clearChunkSelection: () => void
}

// One item of /api/segmentation/tsdf/list — per-instance entries carry
// meta.instance_id, the whole-scene meshes (scene / scene_poisson) don't.
interface TsdfListEntry {
    folder: string
    glb_url: string
    meta: { instance_id?: number; label?: string; method?: string } | null
}

// Vertex shader — matches FusionRenderer.js point size formula
// 256×1 single-channel lookup texture backing uSegVisTex (255 = visible).
// Mutated in place by setSegmentVisibility (write texel + needsUpdate).
const makeSegVisTexture = () => {
    const tex = new THREE.DataTexture(new Uint8Array(256).fill(255), 256, 1,
                                      THREE.RedFormat, THREE.UnsignedByteType)
    tex.minFilter = THREE.NearestFilter
    tex.magFilter = THREE.NearestFilter
    tex.needsUpdate = true
    return tex
}

const vertexShader = `
  attribute float classId;
  attribute float confidence;
  varying float vClassId;
  varying float vConfidence;
  varying float vSegVisible;
  varying vec3 vColor;
  varying vec3 vWorldPos;
  uniform float pointSize;
  // 256 segment-visibility slots via a 256×1 LOOKUP TEXTURE (one texel per
  // instance id). The old float[16] uniform chain capped the viewer at 16
  // instances (test2: 46+). A texel fetch in the vertex shader is the
  // lowest-common-denominator path every driver handles — no dynamic uniform
  // indexing, no uniform-vector pressure.
  uniform sampler2D uSegVisTex;

  void main() {
    vClassId = classId;
    vConfidence = confidence;
    vColor = color;
    vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;

    // Segment visibility lookup (done here because classId attribute is exact)
    if (classId < 0.0) {
      vSegVisible = 1.0; // always visible (e.g. sábana)
    } else {
      float u = (clamp(classId, 0.0, 255.0) + 0.5) / 256.0;
      vSegVisible = texture2D(uSegVisTex, vec2(u, 0.5)).r;
    }

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;

    // Perspective-attenuated size. The *20 factor matters: without it,
    // pointSize/depth at normal viewing distances (~10m) falls below the
    // clamp floor, so the slider had no visible effect. Floor lowered to 0.25
    // so small pointSize values (slider min 0.01) actually render tiny points.
    // NOTE: most GPUs enforce their own ~1px minimum, so sub-pixel sizes may
    // not get smaller than 1px on screen regardless of this floor.
    float depth = -mvPosition.z;
    float size = pointSize * 20.0 / depth;
    gl_PointSize = clamp(size, 0.25, 40.0);
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
  // selection-box highlight (brush box tool): points inside the box being
  // edited light up golden so the selection is visible BEFORE applying
  uniform bool uSelBoxOn;
  uniform mat4 uSelBoxInv;
  // confidence-filter preview (brush): points BELOW this threshold light up
  // red — they would move to unsegmented on apply. -1.0 = off.
  uniform float uConfHl;

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

    if (uSelBoxOn) {
      vec3 lp = (uSelBoxInv * vec4(vWorldPos, 1.0)).xyz;
      if (abs(lp.x) <= 1.0 && abs(lp.y) <= 1.0 && abs(lp.z) <= 1.0) {
        finalColor = mix(finalColor, vec3(1.0, 0.82, 0.2), 0.65);
      }
    }

    if (uConfHl >= 0.0 && vConfidence < uConfHl) {
      finalColor = mix(finalColor, vec3(1.0, 0.25, 0.2), 0.7);
    }

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

// Pre-allocated scratch for setCameraToPose — called every video frame during the
// flythrough. Reused so the per-frame camera update allocates nothing (no GC churn).
const _ctpC2W = new THREE.Matrix4()
const _ctpCvToGl = new THREE.Matrix4().makeScale(1, -1, -1)   // constant CV→GL flip
const _ctpGroup = new THREE.Matrix4()
const _ctpPos = new THREE.Vector3()
const _ctpQuat = new THREE.Quaternion()
const _ctpScl = new THREE.Vector3()
const _ctpFwd = new THREE.Vector3()

const Viewport = forwardRef<ViewportHandle, ViewportProps>(function Viewport(
    { pointSize, pointBudget, confidenceThreshold, activeSession, activeTool, showAxes = true, showGrid = true, pipelineRunning = false, onPointCount, onFps, onStatusMessage, onSegments, onPipelineProgress, onBimLoaded, onSabanaLoaded, onHasConfidence, showCameraPoses = true, onHasCameraPoses, onVolumeChanged, onVolumeDeleted, eraseRadius, eraseShape, eraseYawDeg, onEraseRadiusChange, onEraseMarksChanged, onEraseBoxSelected, onEraseLedger, onTsdfReady, onSceneObjectSelected, onSceneObjectsChanged, onChunkSelected, onChunkDelta },
    ref
) {
    const containerRef = useRef<HTMLDivElement>(null)
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
    const sceneRef = useRef<THREE.Scene | null>(null)
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
    const controlsRef = useRef<OrbitControls | null>(null)
    const pointCloudRef = useRef<THREE.Points | null>(null)
    const materialRef = useRef<THREE.ShaderMaterial | null>(null)
    // Cloud-hidden flag — driven by App (`setCloudObjectVisible`) and by the
    // confidence-slider useEffect. Read by the Potree LOD gate to short-
    // circuit `updateVisibility()` when the cloud is off. We never set
    // `octreeGroup.visible = false` because mesh groups (TSDF/Shape/Recon)
    // live under it for floor_transform inheritance and `visible` is
    // hierarchical in three.js — hiding the parent would drag the meshes too.
    const cloudHiddenRef = useRef(false)
    const animFrameRef = useRef<number>(0)
    const wsRef = useRef<WebSocket | null>(null)
    const totalPointsRef = useRef(0)
    const geometryRef = useRef<THREE.BufferGeometry | null>(null)
    const obbGroupRef = useRef<THREE.Group | null>(null)
    // Shape (ShapeR) meshes — one per segmented instance. Lives as a child of
    // the Potree octreeGroup so floor_transform / alignment edits apply
    // automatically. Cleared/recreated when the session changes.
    const shapesGroupRef = useRef<THREE.Group | null>(null)
    const shapesByInstanceRef = useRef<Map<number, THREE.Group>>(new Map())
    // TSDF meshes — same lifecycle as shapes, kept in a parallel group so both
    // backends can be displayed simultaneously for A/B comparison.
    const tsdfGroupRef = useRef<THREE.Group | null>(null)
    // Lazy TSDF: per-instance meshes are NOT downloaded at session open (only
    // the whole-scene mesh is). Entries wait here (folder → list item) until
    // setTsdfVisibility(folder, true) pulls them in on demand.
    const tsdfPendingRef = useRef<Map<string, TsdfListEntry>>(new Map())
    const tsdfLoadingRef = useRef<Set<string>>(new Set())
    // Reconstruction-v2 scene — typed elements (parametric surfaces / swept solids
    // / boxes / linear-repeats + free-form ShapeR meshes). Same lifecycle as
    // shapes; lives under the octreeGroup so floor_transform applies.
    const reconSceneGroupRef = useRef<THREE.Group | null>(null)
    const reconByInstanceRef = useRef<Map<number, THREE.Group>>(new Map())
    const gltfLoaderRef = useRef<GLTFLoader | null>(null)
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

    // Immersive assistant visualization (animated measurements + user volumes)
    const assistantVizRef = useRef<AssistantViz | null>(null)
    const camTweenRef = useRef<number | null>(null)

    // ── chunk boxes (USER 2026-09-06): one floor-aligned OBB per chunk, all
    // deselected by default; the selected box carries a translate/rotate
    // gizmo (NO scale) centered on the chunk. Boxes live under the potree
    // octreeGroup so the floor transform applies — gizmo deltas measured in
    // the group's LOCAL frame are therefore in the cloud FILE frame.
    const chunkBoxRootRef = useRef<THREE.Group | null>(null)
    const chunkBoxByIdRef = useRef<Map<number, THREE.Group>>(new Map())
    const chunkOrigRef = useRef<Map<number, THREE.Matrix4>>(new Map())
    const [selChunk, setSelChunk] = useState<number | null>(null)
    const [chunkMode, setChunkMode] = useState<'translate' | 'rotate'>('translate')
    const chunkTcRef = useRef<TransformControls | null>(null)
    const onChunkSelectedRef = useRef(onChunkSelected)
    const onChunkDeltaRef = useRef(onChunkDelta)
    useEffect(() => { onChunkSelectedRef.current = onChunkSelected }, [onChunkSelected])
    useEffect(() => { onChunkDeltaRef.current = onChunkDelta }, [onChunkDelta])
    useEffect(() => { onChunkSelectedRef.current?.(selChunk) }, [selChunk])

    const chunkColor = (c: number) => new THREE.Color().setHSL((c * 0.618034) % 1, 0.75, 0.55)

    const emitChunkDelta = (chunk: number) => {
        const g = chunkBoxByIdRef.current.get(chunk)
        const orig = chunkOrigRef.current.get(chunk)
        if (!g || !orig) return
        g.updateMatrix()
        const delta = g.matrix.clone().multiply(orig.clone().invert())
        const identity = delta.elements.every((v, i) =>
            Math.abs(v - (i % 5 === 0 ? 1 : 0)) < 1e-6)
        if (identity) { onChunkDeltaRef.current?.(chunk, null); return }
        const e = delta.elements   // column-major → row-major for the API
        onChunkDeltaRef.current?.(chunk, [
            e[0], e[4], e[8], e[12],
            e[1], e[5], e[9], e[13],
            e[2], e[6], e[10], e[14],
            e[3], e[7], e[11], e[15],
        ])
    }

    // gizmo lifecycle for the selected chunk box
    useEffect(() => {
        const scene = sceneRef.current, camera = cameraRef.current
        const renderer = rendererRef.current, controls = controlsRef.current
        if (!scene || !camera || !renderer || !controls || selChunk == null) return
        const g = chunkBoxByIdRef.current.get(selChunk)
        if (!g) { setSelChunk(null); return }
        const tc = new TransformControls(camera, renderer.domElement)
        tc.setMode(chunkMode)          // translate | rotate ONLY — no scale
        tc.setSize(0.9)
        tc.attach(g)
        scene.add(tc.getHelper())
        tc.addEventListener('dragging-changed', (e) => {
            const dragging = !!(e as unknown as { value?: boolean }).value
            controls.enabled = !dragging
            if (!dragging && selChunk != null) emitChunkDelta(selChunk)
        })
        chunkTcRef.current = tc
        return () => {
            scene.remove(tc.getHelper())
            tc.detach()
            tc.dispose()
            chunkTcRef.current = null
            controls.enabled = true
        }
    }, [selChunk, chunkMode])

    // click-select chunk boxes (navigate tool only; drags never select)
    useEffect(() => {
        const container = containerRef.current
        if (!container) return
        let down: [number, number] | null = null
        const onDown = (e: MouseEvent) => { if (e.button === 0) down = [e.clientX, e.clientY] }
        const onClick = (e: MouseEvent) => {
            if (activeToolRef.current !== 'navigate') return
            if (down && (Math.abs(e.clientX - down[0]) > 4 || Math.abs(e.clientY - down[1]) > 4)) return
            const camera = cameraRef.current, renderer = rendererRef.current
            const root = chunkBoxRootRef.current
            if (!camera || !renderer || !root || !root.children.length) return
            const rect = renderer.domElement.getBoundingClientRect()
            const mouse = new THREE.Vector2(
                ((e.clientX - rect.left) / rect.width) * 2 - 1,
                -((e.clientY - rect.top) / rect.height) * 2 + 1)
            const rc = new THREE.Raycaster()
            rc.setFromCamera(mouse, camera)
            const pickables: THREE.Object3D[] = []
            root.children.forEach(g => g.children.forEach(ch => {
                if (ch.userData.chunkPick) pickables.push(ch)
            }))
            const hits = rc.intersectObjects(pickables, false)
            if (hits.length) {
                setSelChunk(hits[0].object.userData.chunkId as number)
            } else if (!(chunkTcRef.current as unknown as { axis?: string } | null)?.axis) {
                setSelChunk(null)
            }
        }
        container.addEventListener('mousedown', onDown)
        container.addEventListener('click', onClick)
        return () => {
            container.removeEventListener('mousedown', onDown)
            container.removeEventListener('click', onClick)
        }
    }, [])

    // ── evaluation-volume gizmo (select → move/rotate/resize, user 2026-08-29)
    const [selVolume, setSelVolume] = useState<number | null>(null)
    const [volMode, setVolMode] = useState<'translate' | 'rotate' | 'scale'>('translate')
    const [volSolid, setVolSolid] = useState(false)
    const volTcRef = useRef<TransformControls | null>(null)
    const onVolumeChangedRef = useRef(onVolumeChanged)
    const onVolumeDeletedRef = useRef(onVolumeDeleted)
    useEffect(() => { onVolumeChangedRef.current = onVolumeChanged }, [onVolumeChanged])
    useEffect(() => { onVolumeDeletedRef.current = onVolumeDeleted }, [onVolumeDeleted])

    // Gizmo lifecycle: attach TransformControls to the selected volume group.
    useEffect(() => {
        const scene = sceneRef.current, camera = cameraRef.current
        const renderer = rendererRef.current, controls = controlsRef.current
        if (!scene || !camera || !renderer || !controls || selVolume == null) return
        const g = assistantVizRef.current?.getVolumeGroup(selVolume)
        if (!g) { setSelVolume(null); return }
        const tc = new TransformControls(camera, renderer.domElement)
        tc.setMode(volMode)
        if (volMode === 'rotate') { tc.showX = false; tc.showZ = false }  // yaw only
        tc.setSize(0.8)
        tc.attach(g)
        scene.add(tc.getHelper())
        tc.addEventListener('dragging-changed', (e) => {
            const dragging = !!(e as unknown as { value?: boolean }).value
            controls.enabled = !dragging
            if (!dragging) {
                // drag finished → persist + re-evaluate collision state
                const params = assistantVizRef.current?.volumeParams(selVolume)
                if (params) onVolumeChangedRef.current?.(params)
            }
        })
        volTcRef.current = tc
        return () => {
            scene.remove(tc.getHelper())
            tc.detach()
            tc.dispose()
            volTcRef.current = null
            controls.enabled = true
        }
    }, [selVolume, volMode])

    // Click-select volumes (navigate tool only; a drag never selects), and
    // Escape / Delete keyboard handling while one is selected.
    useEffect(() => {
        const container = containerRef.current
        if (!container) return
        let down: [number, number] | null = null
        const onDown = (e: MouseEvent) => { if (e.button === 0) down = [e.clientX, e.clientY] }
        const onClick = (e: MouseEvent) => {
            if (activeToolRef.current !== 'navigate') return
            if (down && (Math.abs(e.clientX - down[0]) > 4 || Math.abs(e.clientY - down[1]) > 4)) return
            const camera = cameraRef.current, renderer = rendererRef.current
            const viz = assistantVizRef.current
            if (!camera || !renderer || !viz) return
            const rect = renderer.domElement.getBoundingClientRect()
            const mouse = new THREE.Vector2(
                ((e.clientX - rect.left) / rect.width) * 2 - 1,
                -((e.clientY - rect.top) / rect.height) * 2 + 1)
            const rc = new THREE.Raycaster()
            rc.setFromCamera(mouse, camera)
            const hits = rc.intersectObjects(viz.pickableVolumes(), false)
            if (hits.length) {
                setSelVolume(hits[0].object.userData.volumeId as number)
            } else if (!(volTcRef.current as unknown as { axis?: string } | null)?.axis) {
                // empty click (and not on a gizmo handle) deselects
                setSelVolume(null)
            }
        }
        const onKey = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setSelVolume(null)
        }
        container.addEventListener('mousedown', onDown)
        container.addEventListener('click', onClick)
        window.addEventListener('keydown', onKey)
        return () => {
            container.removeEventListener('mousedown', onDown)
            container.removeEventListener('click', onClick)
            window.removeEventListener('keydown', onKey)
        }
    }, [])

    // ── Placed library objects (USER 2026-08-31): GLBs from the collection /
    // any project inserted into THIS scene as references — gizmo-alignable,
    // persisted per session (matrix), removable without ever touching the
    // source GLB. ─────────────────────────────────────────────────────────
    const sceneObjectsGroupRef = useRef<THREE.Group | null>(null)
    const sceneObjByIdRef = useRef<Map<number, THREE.Group>>(new Map())
    const [selSceneObj, setSelSceneObj] = useState<number | null>(null)
    const [sceneObjMode, setSceneObjMode] = useState<'translate' | 'rotate' | 'scale'>('translate')
    const sceneTcRef = useRef<TransformControls | null>(null)
    const selSceneObjRef = useRef<number | null>(null)
    useEffect(() => { selSceneObjRef.current = selSceneObj }, [selSceneObj])
    const onSceneObjectSelectedRef = useRef(onSceneObjectSelected)
    useEffect(() => { onSceneObjectSelectedRef.current = onSceneObjectSelected }, [onSceneObjectSelected])
    useEffect(() => { onSceneObjectSelectedRef.current?.(selSceneObj) }, [selSceneObj])

    const onSceneObjectsChangedRef = useRef(onSceneObjectsChanged)
    useEffect(() => { onSceneObjectsChangedRef.current = onSceneObjectsChanged }, [onSceneObjectsChanged])
    const _emitSceneObjects = () => {
        const out: Array<{ id: number; name: string; visible: boolean }> = []
        sceneObjByIdRef.current.forEach((g, id) => out.push({
            id, name: String(g.userData.sceneObjectName || `object ${id}`),
            visible: g.visible }))
        out.sort((a, b) => a.id - b.id)
        onSceneObjectsChangedRef.current?.(out)
    }
    const _removeSceneObject = async (id: number) => {
        const sid = activeSessionRef.current
        if (sid == null) return
        const g = sceneObjByIdRef.current.get(id)
        if (selSceneObjRef.current === id) setSelSceneObj(null)
        if (g) { g.parent?.remove(g); sceneObjByIdRef.current.delete(id) }
        _emitSceneObjects()
        try {
            await fetch('/api/objects/scene/remove', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sid, id }),
            })
            if (onStatusMessage) onStatusMessage('📦 reference removed from the scene (source GLB untouched)')
        } catch { /* silent */ }
    }
    const _sceneObjectsGroup = () => {
        const scene = sceneRef.current
        if (!scene) return null
        let g = sceneObjectsGroupRef.current
        if (!g || g.parent !== scene) {
            g = new THREE.Group()
            g.name = 'scene-objects'
            scene.add(g)
            sceneObjectsGroupRef.current = g
        }
        return g
    }

    const _persistSceneObj = async (id: number) => {
        const sid = activeSessionRef.current
        const g = sceneObjByIdRef.current.get(id)
        if (!sid || !g) return
        g.updateMatrixWorld(true)
        try {
            await fetch('/api/objects/scene/update', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sid, id, matrix: g.matrixWorld.toArray() }),
            })
        } catch { /* silent */ }
    }

    const _loadPlacedObject = async (entry: { id: number; name: string; url: string; matrix?: number[] }, select = false) => {
        const parent = _sceneObjectsGroup()
        if (!parent) return
        if (!gltfLoaderRef.current) gltfLoaderRef.current = makeGltfLoader()
        try {
            const gltf = await gltfLoaderRef.current.loadAsync(entry.url)
            const g = new THREE.Group()
            g.name = `placed-${entry.id}`
            g.userData.sceneObjectId = entry.id
            g.userData.sceneObjectName = entry.name
            // pivot AT THE OBJECT'S CENTER (user 2026-08-31): GLB geometry
            // lives in its source-session coordinates, so the group origin
            // landed far from the mesh and the gizmo was unusable. The child
            // is shifted so the group origin == geometric center; the saved
            // matrix maps that CENTERED object into this scene.
            const inner = gltf.scene
            const bb = new THREE.Box3().setFromObject(inner)
            const ctr = bb.getCenter(new THREE.Vector3())
            inner.position.sub(ctr)
            g.add(inner)
            if (entry.matrix && entry.matrix.length === 16) {
                const M = new THREE.Matrix4().fromArray(entry.matrix)
                M.decompose(g.position, g.quaternion, g.scale)
            } else {
                g.position.copy(ctr)   // fresh insert: same world pose as the source
            }
            parent.add(g)
            sceneObjByIdRef.current.set(entry.id, g)
            _emitSceneObjects()
            if (select) {
                // fresh insert with identity matrix → drop it at the orbit target
                if (!entry.matrix) {
                    const tgt = controlsRef.current?.target
                    if (tgt) { g.position.copy(tgt); _persistSceneObj(entry.id) }
                }
                setSelSceneObj(entry.id)
            }
            if (onStatusMessage && select) onStatusMessage(`📦 "${entry.name}" placed — drag the gizmo to align (G/R keys, Del removes the reference)`)
        } catch (e) {
            console.warn('[Viewport] placed object load failed', entry.url, e)
            if (onStatusMessage) onStatusMessage(`📦 failed to load "${entry.name}"`)
        }
    }

    const reloadSceneObjectsRef = useRef<(sid: string) => Promise<void>>(async () => { /* set below */ })
    reloadSceneObjectsRef.current = async (sid: string) => {
        const parent = _sceneObjectsGroup()
        if (!parent) return
        setSelSceneObj(null)
        for (const g of [...parent.children]) parent.remove(g)
        sceneObjByIdRef.current.clear()
        _emitSceneObjects()
        try {
            const r = await fetch(`/api/objects/scene/${sid}`)
            if (!r.ok) return
            const d = await r.json()
            for (const o of (d.objects || [])) await _loadPlacedObject(o, false)
            if ((d.objects || []).length && onStatusMessage)
                onStatusMessage(`📦 ${(d.objects || []).length} placed object(s) restored`)
        } catch { /* silent */ }
    }

    // gizmo lifecycle for the selected placed object (mirror of volumes)
    useEffect(() => {
        const scene = sceneRef.current, camera = cameraRef.current
        const renderer = rendererRef.current, controls = controlsRef.current
        if (!scene || !camera || !renderer || !controls || selSceneObj == null) return
        const g = sceneObjByIdRef.current.get(selSceneObj)
        if (!g) { setSelSceneObj(null); return }
        const tc = new TransformControls(camera, renderer.domElement)
        tc.setMode(sceneObjMode)
        tc.setSize(0.9)
        tc.attach(g)
        scene.add(tc.getHelper())
        tc.addEventListener('dragging-changed', (e) => {
            const dragging = !!(e as unknown as { value?: boolean }).value
            controls.enabled = !dragging
            if (!dragging && selSceneObj != null) _persistSceneObj(selSceneObj)
        })
        sceneTcRef.current = tc
        return () => {
            scene.remove(tc.getHelper())
            tc.detach()
            tc.dispose()
            sceneTcRef.current = null
            controls.enabled = true
        }
    }, [selSceneObj, sceneObjMode])

    // click-select placed objects (navigate tool; drag never selects)
    useEffect(() => {
        const container = containerRef.current
        if (!container) return
        let down: [number, number] | null = null
        const onDown = (e: MouseEvent) => { if (e.button === 0) down = [e.clientX, e.clientY] }
        const onClick = (e: MouseEvent) => {
            if (activeToolRef.current !== 'navigate') return
            if (down && (Math.abs(e.clientX - down[0]) > 4 || Math.abs(e.clientY - down[1]) > 4)) return
            const camera = cameraRef.current, renderer = rendererRef.current
            const parent = sceneObjectsGroupRef.current
            if (!camera || !renderer || !parent || !parent.children.length) return
            const rect = renderer.domElement.getBoundingClientRect()
            const mouse = new THREE.Vector2(
                ((e.clientX - rect.left) / rect.width) * 2 - 1,
                -((e.clientY - rect.top) / rect.height) * 2 + 1)
            const rc = new THREE.Raycaster()
            rc.setFromCamera(mouse, camera)
            const hits = rc.intersectObjects(parent.children, true)
            if (hits.length) {
                let o: THREE.Object3D | null = hits[0].object
                while (o && o.userData.sceneObjectId === undefined) o = o.parent
                if (o) { setSelSceneObj(o.userData.sceneObjectId as number); return }
            } else if (selSceneObj != null
                       && !(sceneTcRef.current as unknown as { axis?: string } | null)?.axis) {
                setSelSceneObj(null)
            }
        }
        const onKey = (e: KeyboardEvent) => {
            if (selSceneObj == null) return
            if (e.key === 'Escape') setSelSceneObj(null)
            if (e.key === 'g' || e.key === 'G') setSceneObjMode('translate')
            if (e.key === 'r' || e.key === 'R') setSceneObjMode('rotate')
        }
        container.addEventListener('mousedown', onDown)
        container.addEventListener('click', onClick)
        window.addEventListener('keydown', onKey)
        return () => {
            container.removeEventListener('mousedown', onDown)
            container.removeEventListener('click', onClick)
            window.removeEventListener('keydown', onKey)
        }
    }, [selSceneObj])

    // Smooth camera flight (assistant measurements / frameBox): ease-in-out the
    // orbit target and camera position toward the goal instead of snapping.
    // Cancelled the moment the user grabs the controls (see 'start' listener).
    const animateCameraTo = useCallback((toTarget: THREE.Vector3,
                                         toPos: THREE.Vector3, dur = 950) => {
        const camera = cameraRef.current, controls = controlsRef.current
        if (!camera || !controls) return
        if (camTweenRef.current != null) cancelAnimationFrame(camTweenRef.current)
        const fromTarget = controls.target.clone()
        const fromPos = camera.position.clone()
        const t0 = performance.now()
        const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)
        const step = (now: number) => {
            const t = Math.min(1, (now - t0) / dur)
            const k = ease(t)
            controls.target.lerpVectors(fromTarget, toTarget, k)
            camera.position.lerpVectors(fromPos, toPos, k)
            camTweenRef.current = t < 1 ? requestAnimationFrame(step) : null
        }
        camTweenRef.current = requestAnimationFrame(step)
    }, [])

    // Measurement state
    const measureGroupRef = useRef<THREE.Group | null>(null)
    const pendingPointsRef = useRef<THREE.Vector3[]>([])
    const pendingMarkersRef = useRef<THREE.Mesh[]>([])
    const pendingLineRef = useRef<THREE.Line | null>(null)
    const livePreviewRef = useRef<THREE.Group | null>(null)
    const measurementsRef = useRef<Measurement[]>([])
    const raycasterRef = useRef(new THREE.Raycaster())
    const activeToolRef = useRef(activeTool)
    // Eraser tool (user 2026-08-29): brush radius + cursor sphere. The radius
    // is owned by App (slider in the toolbar); the wheel changes it too and
    // reports back so slider and wheel stay in sync.
    const eraseRadiusRef = useRef(0.15)
    const eraseShapeRef = useRef<'sphere' | 'cube' | 'box'>('sphere')
    useEffect(() => { eraseShapeRef.current = (eraseShape === 'cube' || eraseShape === 'box') ? eraseShape : 'sphere' }, [eraseShape])
    // cube yaw about the vertical axis (user 2026-08-30: adapt to diagonal walls)
    const eraseYawRef = useRef(0)
    useEffect(() => {
        eraseYawRef.current = ((eraseYawDeg ?? 0) * Math.PI) / 180
        const cur = eraseCursorRef.current
        if (cur && cur.userData.shape === 'cube') cur.rotation.y = eraseYawRef.current
    }, [eraseYawDeg])
    const eraseCursorRef = useRef<THREE.Mesh | null>(null)
    const onEraseRadiusChangeRef = useRef(onEraseRadiusChange)
    useEffect(() => { onEraseRadiusChangeRef.current = onEraseRadiusChange }, [onEraseRadiusChange])
    const onEraseMarksChangedRef = useRef(onEraseMarksChanged)
    useEffect(() => { onEraseMarksChangedRef.current = onEraseMarksChanged }, [onEraseMarksChanged])
    // mark/commit API installed by the main effect (marks live in its closure)
    const eraseApiRef = useRef<{ commit: (target?: number | null, newLabel?: string, onlyInstances?: number[], includeUnsegmented?: boolean) => Promise<void>; clear: () => void; confApply: (thr: number, onlyInstances?: number[], includeUnsegmented?: boolean) => Promise<void> } | null>(null)
    const eraseBoxApiRef = useRef<{ setMode: (m: 'translate' | 'rotate' | 'scale') => void; remove: () => void } | null>(null)
    const onEraseBoxSelectedRef = useRef(onEraseBoxSelected)
    useEffect(() => { onEraseBoxSelectedRef.current = onEraseBoxSelected }, [onEraseBoxSelected])
    const onEraseLedgerRef = useRef(onEraseLedger)
    useEffect(() => { onEraseLedgerRef.current = onEraseLedger }, [onEraseLedger])
    const onTsdfReadyRef = useRef(onTsdfReady)
    useEffect(() => { onTsdfReadyRef.current = onTsdfReady }, [onTsdfReady])
    // bridge: renderOBBs is declared later in the file — the potree_ready
    // handler needs it to resync OBBs + panel after an erase/refresh rebuild
    const renderOBBsRef = useRef<((instances: Array<Record<string, unknown>>) => void) | null>(null)
    // user-chosen per-segment point visibility (user 2026-08-30: an OBB/panel
    // resync must NOT resurrect segments the user had hidden)
    const segVisRef = useRef<Map<number, boolean>>(new Map())
    // Raycast honesty (user 2026-08-30): three.js raycasts hit EVERY point in
    // a node's geometry, including points the shader hides — the brush/measure
    // cursor landed on invisible points of hidden segments in MIXED nodes.
    // A hit on a point whose class is hidden in the panel does not count.
    const hitIsVisible = useCallback((hit: THREE.Intersection): boolean => {
        const obj = hit.object as THREE.Points
        if (!(obj as unknown as { isPoints?: boolean }).isPoints) return true
        if (hit.index === undefined || hit.index === null) return true
        const attr = (obj.geometry as THREE.BufferGeometry).getAttribute('classId')
        if (!attr) return true
        const cls = attr.getX(hit.index)
        if (cls < 0) return true
        return segVisRef.current.get(cls) !== false
    }, [])
    useEffect(() => {
        if (activeTool !== 'erase') eraseApiRef.current?.clear()
    }, [activeTool])
    useEffect(() => {
        if (typeof eraseRadius === 'number' && eraseRadius > 0) {
            eraseRadiusRef.current = eraseRadius
            eraseCursorRef.current?.scale.setScalar(eraseRadius)
        }
    }, [eraseRadius])
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
    // WebGL context loss (GPU out of resources / driver reset). Without this
    // the canvas silently freezes or goes blank — the overlay tells the user
    // what happened and how to recover.
    const [contextLost, setContextLost] = useState(false)

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

        // Raycast targets. Mesh groups are always candidates so measurements
        // snap to whatever the user is seeing. Cloud points are added ONLY
        // when the cloud is not hidden — three.js's raycaster does NOT skip
        // invisible objects automatically, so without this filter the
        // (invisible) points still grab the hit, floating just above the mesh.
        const targets: THREE.Object3D[] = []
        if (!cloudHiddenRef.current) {
            if (pointCloud) targets.push(pointCloud)
            if (potreeLoaderRef.current) {
                const octreeGroup = sceneRef.current?.getObjectByName('potree-octree')
                if (octreeGroup) {
                    octreeGroup.children.forEach(c => {
                        if ((c.name || '').startsWith('potree-node-') && c.visible) {
                            targets.push(c)
                        }
                    })
                }
            }
        }
        if (tsdfGroupRef.current) targets.push(tsdfGroupRef.current)
        if (shapesGroupRef.current) targets.push(shapesGroupRef.current)
        if (reconSceneGroupRef.current) targets.push(reconSceneGroupRef.current)
        if (bimGroupRef.current) targets.push(bimGroupRef.current)
        // evaluation volumes are measurable like the scene (user 2026-08-29)
        if (assistantVizRef.current) targets.push(...assistantVizRef.current.pickableVolumes())
        const intersects = raycaster.intersectObjects(targets)
        // Filter out section-box-clipped points AND shader-hidden points
        const hit = intersects.find(i => {
            if (!hitIsVisible(i)) return false
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

        // Raycast targets. Mesh groups are always candidates so measurements
        // snap to whatever the user is seeing. Cloud points are added ONLY
        // when the cloud is not hidden — three.js's raycaster does NOT skip
        // invisible objects automatically, so without this filter the
        // (invisible) points still grab the hit, floating just above the mesh.
        const targets: THREE.Object3D[] = []
        if (!cloudHiddenRef.current) {
            if (pointCloud) targets.push(pointCloud)
            if (potreeLoaderRef.current) {
                const octreeGroup = sceneRef.current?.getObjectByName('potree-octree')
                if (octreeGroup) {
                    octreeGroup.children.forEach(c => {
                        if ((c.name || '').startsWith('potree-node-') && c.visible) {
                            targets.push(c)
                        }
                    })
                }
            }
        }
        if (tsdfGroupRef.current) targets.push(tsdfGroupRef.current)
        if (shapesGroupRef.current) targets.push(shapesGroupRef.current)
        if (reconSceneGroupRef.current) targets.push(reconSceneGroupRef.current)
        if (bimGroupRef.current) targets.push(bimGroupRef.current)
        // evaluation volumes are measurable like the scene (user 2026-08-29)
        if (assistantVizRef.current) targets.push(...assistantVizRef.current.pickableVolumes())
        const intersects = raycaster.intersectObjects(targets)
        // Filter out section-box-clipped points AND shader-hidden points
        const hit = intersects.find(i => {
            if (!hitIsVisible(i)) return false
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
    // Deep GPU disposal for mesh groups: material.dispose() does NOT free the
    // textures it references — the texrecon atlases (tens of MB each) leaked
    // on every clear/reload and the viewer grew heavier with use (2026-08-30)
    const disposeMeshGroup = useCallback((root: THREE.Object3D) => {
        root.traverse((obj) => {
            const mesh = obj as THREE.Mesh
            if (!(mesh as unknown as { isMesh?: boolean }).isMesh) return
            mesh.geometry?.dispose()
            const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
            for (const m of mats) {
                if (!m) continue
                const anyM = m as unknown as Record<string, { dispose?: () => void } | undefined>
                for (const k of ['map', 'alphaMap', 'aoMap', 'emissiveMap', 'normalMap', 'roughnessMap', 'metalnessMap']) {
                    anyM[k]?.dispose?.()
                }
                m.dispose()
            }
        })
    }, [])

    // ── ShapeR mesh auto-load helpers ────────────────────────────────
    const clearAllShapes = useCallback(() => {
        const group = shapesGroupRef.current
        if (group) {
            disposeMeshGroup(group)   // geometry + materials + TEXTURES
            const parent = group.parent
            if (parent) parent.remove(group)
        }
        shapesGroupRef.current = null
        shapesByInstanceRef.current.clear()
    }, [])

    const loadShapesIntoGroup = useCallback(async (sessionId: string, parentGroup: THREE.Object3D) => {
        // Reset previous shapes (different session or refresh)
        clearAllShapes()

        let res: Response
        try {
            res = await fetch(`/api/segmentation/shape/list/${sessionId}`)
        } catch (e) {
            console.warn('[Viewport] shape list fetch failed', e)
            return
        }
        if (!res.ok) return
        const data = await res.json()
        const shapes = (data?.shapes || []) as Array<{ folder: string; glb_url: string; meta: any }>
        if (shapes.length === 0) return

        const group = new THREE.Group()
        group.name = 'shapes-group'
        parentGroup.add(group)
        shapesGroupRef.current = group

        if (!gltfLoaderRef.current) gltfLoaderRef.current = makeGltfLoader()
        const loader = gltfLoaderRef.current

        let loaded = 0
        for (const sh of shapes) {
            try {
                const gltf = await loader.loadAsync(sh.glb_url)
                const meshGroup = new THREE.Group()
                meshGroup.name = `shape-${sh.folder}`
                meshGroup.userData = { meta: sh.meta, folder: sh.folder }
                meshGroup.add(gltf.scene)
                group.add(meshGroup)

                const iid = sh.meta?.instance_id
                if (typeof iid === 'number') {
                    shapesByInstanceRef.current.set(iid, meshGroup)
                }
                loaded += 1
            } catch (e) {
                console.warn(`[Viewport] failed to load mesh ${sh.glb_url}`, e)
            }
        }
        if (onStatusMessage && loaded > 0) {
            onStatusMessage(`Loaded ${loaded} reconstructed mesh${loaded !== 1 ? 'es' : ''}`)
        }
    }, [clearAllShapes, onStatusMessage])

    // ── TSDF mesh auto-load helpers (mirrors ShapeR — separate group) ──
    const clearAllTsdf = useCallback(() => {
        const group = tsdfGroupRef.current
        if (group) {
            disposeMeshGroup(group)   // geometry + materials + TEXTURES
            const parent = group.parent
            if (parent) parent.remove(group)
        }
        tsdfGroupRef.current = null
        tsdfPendingRef.current.clear()
        tsdfLoadingRef.current.clear()
    }, [])

    /** Download + attach one TSDF GLB under the tsdf-group. Shared by the
     *  eager scene-mesh load and the on-demand per-instance load. */
    const loadTsdfMeshEntry = useCallback(async (sh: TsdfListEntry): Promise<THREE.Group | null> => {
        const group = tsdfGroupRef.current
        if (!group) return null
        if (!gltfLoaderRef.current) gltfLoaderRef.current = makeGltfLoader()
        const loader = gltfLoaderRef.current

        const gltf = await loader.loadAsync(sh.glb_url)
        // TSDF normals point INTO the room (marching cubes uses the
        // SDF gradient, which points from the wall toward free space —
        // and free space is the interior, since scanning happens from
        // inside). With FrontSide (three.js default) walls disappear
        // when seen from outside. DoubleSide renders both faces so the
        // mesh is navigable from any angle. ~2× rasterized triangles,
        // negligible at this tri count.
        gltf.scene.traverse((obj) => {
            if (obj instanceof THREE.Mesh) {
                const mat = obj.material
                if (Array.isArray(mat)) {
                    mat.forEach(m => { if (m) m.side = THREE.DoubleSide })
                } else if (mat) {
                    mat.side = THREE.DoubleSide
                }
            }
        })
        // Session may have switched while the GLB was downloading — drop it
        if (tsdfGroupRef.current !== group) return null
        const meshGroup = new THREE.Group()
        meshGroup.name = `tsdf-${sh.folder}`
        meshGroup.userData = { meta: sh.meta, folder: sh.folder, method: 'tsdf' }
        meshGroup.add(gltf.scene)
        group.add(meshGroup)
        return meshGroup
    }, [])

    const loadTsdfIntoGroup = useCallback(async (sessionId: string, parentGroup: THREE.Object3D) => {
        clearAllTsdf()
        let res: Response
        try {
            res = await fetch(`/api/segmentation/tsdf/list/${sessionId}`)
        } catch (e) {
            console.warn('[Viewport] tsdf list fetch failed', e)
            return
        }
        if (!res.ok) return
        const data = await res.json()
        const meshes = (data?.shapes || []) as TsdfListEntry[]
        if (meshes.length === 0) return

        const group = new THREE.Group()
        group.name = 'tsdf-group'
        parentGroup.add(group)
        tsdfGroupRef.current = group

        // Lazy split: only whole-scene meshes (no instance_id — scene /
        // scene_poisson) download at session open. Eagerly pulling every
        // per-instance GLB blew up big sessions (test2: 40 GLBs ≈ 230 MB on
        // top of the point cloud → renderer OOM). Instance meshes wait in
        // tsdfPendingRef until setTsdfVisibility(folder, true) requests them.
        const eager: TsdfListEntry[] = []
        for (const sh of meshes) {
            if (typeof sh.meta?.instance_id === 'number') tsdfPendingRef.current.set(sh.folder, sh)
            else eager.push(sh)
        }

        let loaded = 0
        for (const sh of eager) {
            try {
                if (await loadTsdfMeshEntry(sh)) loaded += 1
            } catch (e) {
                console.warn(`[Viewport] failed to load TSDF mesh ${sh.glb_url}`, e)
            }
        }
        if (onStatusMessage && loaded > 0) {
            const pending = tsdfPendingRef.current.size
            onStatusMessage(`Loaded ${loaded} TSDF scene mesh${loaded !== 1 ? 'es' : ''}${pending > 0 ? ` — ${pending} instance mesh${pending !== 1 ? 'es' : ''} on demand` : ''}`)
        }
    }, [clearAllTsdf, loadTsdfMeshEntry, onStatusMessage])

    // ── Reconstruction-v2 scene loader (mirrors ShapeR / TSDF; own group) ──
    const clearAllReconScene = useCallback(() => {
        const group = reconSceneGroupRef.current
        if (group) {
            disposeMeshGroup(group)   // geometry + materials + TEXTURES
            const parent = group.parent
            if (parent) parent.remove(group)
        }
        reconSceneGroupRef.current = null
        reconByInstanceRef.current.clear()
    }, [])

    const loadReconSceneIntoGroup = useCallback(async (
        sessionId: string,
        parentGroup: THREE.Object3D,
        preloaded?: { elements?: Array<any> } | null,
    ) => {
        clearAllReconScene()
        let elements: Array<any>
        if (preloaded && Array.isArray(preloaded.elements)) {
            // ``potree_ready`` already bundles the scene payload — skip the fetch.
            elements = preloaded.elements
        } else {
            let res: Response
            try {
                res = await fetch(`/api/segmentation/scene/${sessionId}`)
            } catch (e) {
                console.warn('[Viewport] scene fetch failed', e)
                return
            }
            if (!res.ok) return
            const data = await res.json()
            if (!data?.exists) return
            elements = (data?.elements || []) as Array<any>
        }
        if (elements.length === 0) return

        const group = new THREE.Group()
        group.name = 'recon-scene-group'
        parentGroup.add(group)
        reconSceneGroupRef.current = group

        if (!gltfLoaderRef.current) gltfLoaderRef.current = makeGltfLoader()
        const loader = gltfLoaderRef.current

        // confidence colours (mesh elements carry a per-vertex `observed` attribute)
        const OBS_GREEN = new THREE.Color(0.20, 0.85, 0.32)
        const PRED_AMBER = new THREE.Color(0.95, 0.74, 0.20)

        let loaded = 0
        for (const el of elements) {
            const url: string | null = el?.glb_url
            const iid: number | undefined = el?.instance_id
            if (!url) continue
            try {
                const gltf = await loader.loadAsync(url)
                const meshGroup = new THREE.Group()
                meshGroup.name = `recon-${el?.label || iid}`
                meshGroup.userData = { meta: el, kind: el?.kind, geometryClass: el?.geometry_class }
                // colour mesh-element vertices by the `observed` confidence attr
                gltf.scene.traverse((obj) => {
                    if (!(obj instanceof THREE.Mesh)) return
                    const geo = obj.geometry as THREE.BufferGeometry
                    const obsAttr = (geo.getAttribute('_observed') || geo.getAttribute('observed')) as THREE.BufferAttribute | undefined
                    if (obsAttr && obsAttr.count === geo.getAttribute('position')?.count) {
                        const n = obsAttr.count
                        const colors = new Float32Array(n * 3)
                        for (let i = 0; i < n; i++) {
                            const c = obsAttr.getX(i) > 0.5 ? OBS_GREEN : PRED_AMBER
                            colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b
                        }
                        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
                        const mat = (Array.isArray(obj.material) ? obj.material[0] : obj.material) as THREE.Material
                        const cloned = mat.clone() as any
                        cloned.vertexColors = true
                        obj.material = cloned
                    }
                })
                meshGroup.add(gltf.scene)
                group.add(meshGroup)
                if (typeof iid === 'number') reconByInstanceRef.current.set(iid, meshGroup)
                loaded += 1
            } catch (e) {
                console.warn(`[Viewport] failed to load recon element ${url}`, e)
            }
        }
        if (onStatusMessage && loaded > 0) {
            onStatusMessage(`Loaded reconstruction scene: ${loaded} element${loaded !== 1 ? 's' : ''}`)
        }
    }, [clearAllReconScene, onStatusMessage])

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
        setAssistantObjects: (objects: SceneObject[]) => {
            assistantVizRef.current?.setObjects(objects)
        },
        visualizeMeasurement: (trace: TraceEntry[]) => {
            const box = assistantVizRef.current?.visualizeTrace(trace)
            if (box && !box.isEmpty()) {
                // fly to what was drawn — smooth ease-in-out, cancelled by any
                // user interaction (the geometry reveal animates in parallel)
                const camera = cameraRef.current, controls = controlsRef.current
                if (camera && controls) {
                    const center = box.getCenter(new THREE.Vector3())
                    const radius = Math.max(0.4, box.getSize(new THREE.Vector3()).length() / 2)
                    const dir = camera.position.clone().sub(controls.target).normalize()
                    const dist = radius / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.4
                    animateCameraTo(center, center.clone().addScaledVector(dir, dist))
                }
            }
        },
        clearAssistantViz: () => assistantVizRef.current?.clearMeasurements(),
        addUserVolume: (volume: UserVolume) => assistantVizRef.current?.addVolume(volume),
        removeUserVolume: (volumeId: number) => assistantVizRef.current?.removeVolume(volumeId),
        setVolumeStatus: (volumeId: number, status: 'free' | 'touching' | 'colliding') =>
            assistantVizRef.current?.setVolumeStatus(volumeId, status),
        setVolumeSolid: (volumeId: number, solid: boolean) =>
            assistantVizRef.current?.setVolumeSolid(volumeId, solid),
        setChunkBoxes: (boxes) => {
            const scene = sceneRef.current
            if (!scene) return
            const octree = scene.getObjectByName('potree-octree') as THREE.Group | null
            const parent = octree || scene
            if (!chunkBoxRootRef.current) {
                const root = new THREE.Group()
                root.name = 'chunk-boxes'
                parent.add(root)
                chunkBoxRootRef.current = root
            } else if (chunkBoxRootRef.current.parent !== parent) {
                chunkBoxRootRef.current.parent?.remove(chunkBoxRootRef.current)
                parent.add(chunkBoxRootRef.current)
            }
            const root = chunkBoxRootRef.current
            const want = new Set(boxes.map(b => b.chunk))
            // remove boxes no longer selected
            for (const [cid, g] of chunkBoxByIdRef.current) {
                if (!want.has(cid)) {
                    root.remove(g)
                    chunkBoxByIdRef.current.delete(cid)
                    chunkOrigRef.current.delete(cid)
                    if (selChunk === cid) setSelChunk(null)
                }
            }
            for (const b of boxes) {
                if (chunkBoxByIdRef.current.has(b.chunk)) continue
                const g = new THREE.Group()
                g.position.set(b.center[0], b.center[1], b.center[2])
                g.rotation.y = -b.yaw
                const geo = new THREE.BoxGeometry(b.size[0], b.size[1], b.size[2])
                const col = chunkColor(b.chunk)
                const edges = new THREE.LineSegments(
                    new THREE.EdgesGeometry(geo),
                    new THREE.LineBasicMaterial({ color: col }))
                const fill = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
                    color: col, transparent: true, opacity: 0.05,
                    depthWrite: false, side: THREE.DoubleSide }))
                fill.userData.chunkPick = true
                fill.userData.chunkId = b.chunk
                g.add(edges)
                g.add(fill)
                root.add(g)
                g.updateMatrix()
                chunkBoxByIdRef.current.set(b.chunk, g)
                chunkOrigRef.current.set(b.chunk, g.matrix.clone())
            }
        },
        setChunkGizmoMode: (m) => setChunkMode(m),
        resetChunkBox: (chunk) => {
            const g = chunkBoxByIdRef.current.get(chunk)
            const orig = chunkOrigRef.current.get(chunk)
            if (!g || !orig) return
            orig.decompose(g.position, g.quaternion, g.scale)
            g.updateMatrix()
            onChunkDeltaRef.current?.(chunk, null)
        },
        clearChunkSelection: () => setSelChunk(null),
        frameBox: (min: number[], max: number[]) => {
            const camera = cameraRef.current, controls = controlsRef.current
            if (!camera || !controls) return
            const box = new THREE.Box3(new THREE.Vector3().fromArray(min), new THREE.Vector3().fromArray(max))
            const center = box.getCenter(new THREE.Vector3())
            const radius = Math.max(0.4, box.getSize(new THREE.Vector3()).length() / 2)
            const dir = camera.position.clone().sub(controls.target).normalize()
            const dist = radius / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.4
            animateCameraTo(center, center.clone().addScaledVector(dir, dist))
        },
        setFloorTransform: (arr: number[]) => {
            // Apply a new floor transform (16 floats, column-major) to the
            // octree group at runtime — same path the alignment-gizmo save
            // uses, so cloud + TSDF/shape children update instantly with no
            // session reload. Used by the floor→y=0 leveling flow.
            const loader = potreeLoaderRef.current
            if (!loader || !arr || arr.length !== 16) return
            const M = new THREE.Matrix4().fromArray(arr)
            floorTransformRef.current = M.clone()
            const octreeGroup = loader.getOctreeGroup()
            if (octreeGroup) {
                octreeGroup.matrixAutoUpdate = true
                loader.setTransform(arr)
                octreeGroup.matrixWorldNeedsUpdate = true
            }
        },
        setOBBsVisible: (visible: boolean) => {
            const group = obbGroupRef.current
            if (group) group.visible = visible
        },
        setSegmentVisibility: (segId: number, visible: boolean) => {
            segVisRef.current.set(segId, visible)
            // whole-node culling: octree nodes whose every point is hidden
            // skip the draw entirely (user 2026-08-30: single-segment work
            // was as heavy as the full cloud)
            const hidden = new Set<number>()
            segVisRef.current.forEach((vis, id) => { if (!vis) hidden.add(id) })
            potreeLoaderRef.current?.setClassVisibility(hidden)
            const mat = materialRef.current
            if (!mat) return
            const idx = Math.max(0, Math.min(segId, 255))
            const tex = mat.uniforms.uSegVisTex.value as THREE.DataTexture
            ;(tex.image.data as Uint8Array)[idx] = visible ? 255 : 0
            tex.needsUpdate = true
        },
        setCloudObjectVisible: (visible: boolean) => {
            // Object-level cloud visibility, driven by App. Hide ONLY the
            // Potree-owned point nodes (`potree-node-*` in PotreeLoader) —
            // never the octreeGroup itself, because the mesh groups
            // (TSDF/Shape/Recon) live under it for floor_transform inheritance
            // and `visible` is hierarchical in three.js. cloudHiddenRef also
            // gates the LOD loop (skips updateVisibility + node loading).
            cloudHiddenRef.current = !visible
            const pc = pointCloudRef.current
            if (pc) pc.visible = visible
            const octreeGroup = sceneRef.current?.getObjectByName('potree-octree')
            if (octreeGroup) {
                octreeGroup.children.forEach(child => {
                    if ((child.name || '').startsWith('potree-node-')) {
                        child.visible = visible
                    }
                })
            }
            // re-showing the cloud must NOT resurrect nodes whose segments are
            // hidden — reapply the class-based node culling
            if (visible) {
                const hidden = new Set<number>()
                segVisRef.current.forEach((vis, id) => { if (!vis) hidden.add(id) })
                potreeLoaderRef.current?.setClassVisibility(hidden)
            }
            // Hidden Three.js objects keep their GPU buffers, so visible=false alone
            // does NOT free VRAM — the Potree nodes kept occupying it and slowing the
            // whole scene (incl. the TSDF). Free the GPU geometry on hide; the LOD
            // loop reloads the visible nodes on show.
            const loader = potreeLoaderRef.current
            if (loader) {
                if (!visible) loader.releaseGPUMemory()
                else if (loader.isLoaded) loader.updateVisibility()   // reload promptly
            }
        },
        reloadShapes: async (sessionId: string) => {
            // Reuse current octreeGroup as parent if a cloud is loaded; otherwise
            // attach to the scene root. Either way, floor_transform applies via
            // the parent chain.
            const loader = potreeLoaderRef.current
            const parent = loader?.getOctreeGroup() || sceneRef.current
            if (!parent) return
            await loadShapesIntoGroup(sessionId, parent)
        },
        setShapeVisibility: (instanceId: number, visible: boolean) => {
            const meshGroup = shapesByInstanceRef.current.get(instanceId)
            if (meshGroup) meshGroup.visible = visible
        },
        clearShapes: () => clearAllShapes(),
        reloadTsdf: async (sessionId: string) => {
            const loader = potreeLoaderRef.current
            const parent = loader?.getOctreeGroup() || sceneRef.current
            if (!parent) return
            await loadTsdfIntoGroup(sessionId, parent)
        },
        setTsdfVisibility: (folder: string, visible: boolean) => {
            const meshGroup = tsdfGroupRef.current?.children
                .find(c => c.name === `tsdf-${folder}`)
            if (meshGroup) {
                meshGroup.visible = visible
                return
            }
            // Not in the scene yet — per-instance meshes load on demand the
            // first time they're toggled visible (see loadTsdfIntoGroup).
            if (!visible) return
            const pending = tsdfPendingRef.current.get(folder)
            if (!pending || tsdfLoadingRef.current.has(folder)) return
            tsdfLoadingRef.current.add(folder)
            if (onStatusMessage) onStatusMessage(`Loading TSDF mesh: ${folder}...`)
            loadTsdfMeshEntry(pending).then((g) => {
                tsdfLoadingRef.current.delete(folder)
                if (g) {
                    tsdfPendingRef.current.delete(folder)
                    if (onStatusMessage) onStatusMessage(`TSDF mesh loaded: ${folder}`)
                }
            }).catch((e) => {
                // Keep the entry pending so re-toggling retries the download
                tsdfLoadingRef.current.delete(folder)
                console.warn(`[Viewport] on-demand TSDF load failed for ${folder}`, e)
                if (onStatusMessage) onStatusMessage(`TSDF mesh load failed: ${folder}`)
            })
        },
        clearTsdf: () => clearAllTsdf(),
        reloadReconScene: async (sessionId: string) => {
            const loader = potreeLoaderRef.current
            const parent = loader?.getOctreeGroup() || sceneRef.current
            if (!parent) return
            await loadReconSceneIntoGroup(sessionId, parent)
        },
        setReconElementVisibility: (instanceId: number, visible: boolean) => {
            const g = reconByInstanceRef.current.get(instanceId)
            if (g) g.visible = visible
        },
        clearReconScene: () => clearAllReconScene(),
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
        commitErase: async (target?: number | null, newLabel?: string, onlyInstances?: number[], includeUnsegmented?: boolean) => { await eraseApiRef.current?.commit(target, newLabel, onlyInstances, includeUnsegmented) },
        // ── placed library objects ──
        placeSceneObject: async (entry: { id: number; name: string; url: string; matrix?: number[] }) => {
            await _loadPlacedObject(entry, true)
        },
        reloadSceneObjects: async (sessionId: string) => {
            await reloadSceneObjectsRef.current(sessionId)
        },
        setSceneObjectMode: (m: 'translate' | 'rotate' | 'scale') => setSceneObjMode(m),
        removeSelectedSceneObject: async () => {
            if (selSceneObjRef.current != null) await _removeSceneObject(selSceneObjRef.current)
        },
        removeSceneObject: async (id: number) => { await _removeSceneObject(id) },
        setSceneObjectVisible: (id: number, visible: boolean) => {
            const g = sceneObjByIdRef.current.get(id)
            if (!g) return
            g.visible = visible
            if (!visible && selSceneObjRef.current === id) setSelSceneObj(null)
            _emitSceneObjects()
        },
        getSceneAlignTargets: () => {
            const out: Array<{ key: string; label: string }> = []
            sceneObjByIdRef.current.forEach((g, id) => {
                if (id !== selSceneObjRef.current)
                    out.push({ key: `obj:${id}`, label: String(g.userData.sceneObjectName || `object ${id}`) })
            })
            const tg = tsdfGroupRef.current
            if (tg) for (const c of tg.children)
                if (c.visible && c.name.startsWith('tsdf-'))
                    out.push({ key: `mesh:${c.name.slice(5)}`, label: c.name.slice(5) })
            return out
        },
        alignSceneObject: (op: 'floor' | 'same_base' | 'on_top' | 'center_xz' | 'center_y', targetKey?: string) => {
            const id = selSceneObjRef.current
            if (id == null) return
            const g = sceneObjByIdRef.current.get(id)
            if (!g) return
            g.updateMatrixWorld(true)
            const box = new THREE.Box3().setFromObject(g)
            let tbox: THREE.Box3 | null = null
            if (targetKey) {
                let t: THREE.Object3D | undefined
                if (targetKey.startsWith('obj:'))
                    t = sceneObjByIdRef.current.get(Number(targetKey.slice(4)))
                else if (targetKey.startsWith('mesh:'))
                    t = tsdfGroupRef.current?.children.find(c => c.name === `tsdf-${targetKey.slice(5)}`)
                if (t) { t.updateMatrixWorld(true); tbox = new THREE.Box3().setFromObject(t) }
            }
            if (op === 'floor') g.position.y += -box.min.y
            else if (tbox) {
                if (op === 'same_base') g.position.y += tbox.min.y - box.min.y
                else if (op === 'on_top') g.position.y += tbox.max.y - box.min.y
                else if (op === 'center_xz') {
                    const c = box.getCenter(new THREE.Vector3())
                    const tc2 = tbox.getCenter(new THREE.Vector3())
                    g.position.x += tc2.x - c.x
                    g.position.z += tc2.z - c.z
                } else if (op === 'center_y') {
                    const c = box.getCenter(new THREE.Vector3())
                    const tc2 = tbox.getCenter(new THREE.Vector3())
                    g.position.y += tc2.y - c.y
                }
            }
            _persistSceneObj(id)
        },
        setConfHighlight: (thr: number | null) => {
            const mat = materialRef.current
            if (mat) mat.uniforms.uConfHl.value = thr === null ? -1.0 : thr
        },
        applyConfidenceFilter: async (thr: number, onlyInstances?: number[], includeUnsegmented?: boolean) => {
            await eraseApiRef.current?.confApply(thr, onlyInstances, includeUnsegmented)
            const mat = materialRef.current
            if (mat) mat.uniforms.uConfHl.value = -1.0
        },
        setEraseBoxMode: (m: 'translate' | 'rotate' | 'scale') => eraseBoxApiRef.current?.setMode(m),
        removeSelectedEraseBox: () => eraseBoxApiRef.current?.remove(),
        clearEraseMarks: () => eraseApiRef.current?.clear(),
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
        setFlythroughActive: (active: boolean) => {
            // disable orbit controls while the flythrough drives the camera
            if (controlsRef.current) controlsRef.current.enabled = !active
            // hide the camera-pose markers during playback; restore on close
            // (on normal load they're shown + toggleable). cameraGroupRef holds them.
            if (cameraGroupRef.current) cameraGroupRef.current.visible = !active
            // Save the user's FOV on enter; restore it on exit (setCameraFov sets
            // the real-camera FOV during the flythrough).
            const cam = cameraRef.current
            if (cam) {
                if (active) {
                    if (cam.userData.savedFov == null) cam.userData.savedFov = cam.fov
                } else if (cam.userData.savedFov != null) {
                    cam.fov = cam.userData.savedFov
                    cam.userData.savedFov = null
                    cam.updateProjectionMatrix()
                }
            }
        },
        setCameraFov: (fovYDeg: number) => {
            const cam = cameraRef.current
            if (!cam || !isFinite(fovYDeg) || fovYDeg <= 0 || fovYDeg >= 180) return
            cam.fov = fovYDeg
            cam.updateProjectionMatrix()
        },
        setCameraToPose: (c2wRowMajor: number[]) => {
            const cam = cameraRef.current
            const ctrl = controlsRef.current
            if (!cam || !c2wRowMajor || c2wRowMajor.length < 16) return
            // c2w in the cloud's NATIVE frame (row-major). fromArray expects
            // column-major → transpose. Apply the same group transform used for
            // the displayed cloud (floor align). Convert OpenCV cam convention
            // (+Z forward, +Y down) → three.js (-Z forward, +Y up) via diag(1,-1,-1).
            // NOTE: if the flythrough view comes out mirrored/upside-down, this
            // CV→GL flip is the knob to adjust.
            const c2w = _ctpC2W.fromArray(c2wRowMajor).transpose()
            c2w.multiply(_ctpCvToGl)
            // Apply the SAME transform the displayed cloud uses (floor_transform /
            // alignment) so the camera path lives in the cloud's displayed frame.
            // (Earlier this used cameraGroupRef.matrixWorld — wrong matrix → camera
            // ended up far below. floorTransformRef is what the cloud bbox uses.)
            const groupM = floorTransformRef.current
                ? _ctpGroup.copy(floorTransformRef.current)
                : _ctpGroup.identity()
            const world = groupM.multiply(c2w)
            const pos = _ctpPos, quat = _ctpQuat, scl = _ctpScl
            world.decompose(pos, quat, scl)
            cam.position.copy(pos)
            cam.quaternion.copy(quat)
            cam.updateMatrixWorld(true)
            if (ctrl) {
                const fwd = _ctpFwd.set(0, 0, -1).applyQuaternion(quat)
                ctrl.target.copy(pos).add(fwd.multiplyScalar(2.0))
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

        // Renderer — PBR-ready: sRGB output + ACES filmic tone mapping so the GLBs'
        // PBR materials (concrete / wood / metal / glass / ...) render naturally
        // instead of washed-out / clipped.
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
        renderer.setPixelRatio(window.devicePixelRatio)
        renderer.setSize(container.clientWidth, container.clientHeight)
        renderer.setClearColor(0x0d1117, 1)
        renderer.outputColorSpace = THREE.SRGBColorSpace
        renderer.toneMapping = THREE.ACESFilmicToneMapping
        renderer.toneMappingExposure = 1.0
        container.appendChild(renderer.domElement)
        rendererRef.current = renderer

        // WebGL context loss safety net. preventDefault() tells the browser we
        // want a `webglcontextrestored` — three.js then re-uploads its GPU
        // state from the CPU-side buffers automatically.
        const handleContextLost = (e: Event) => {
            e.preventDefault()
            console.error('[Viewport] WebGL context lost — GPU out of resources or driver reset')
            setContextLost(true)
        }
        const handleContextRestored = () => {
            console.warn('[Viewport] WebGL context restored')
            setContextLost(false)
        }
        renderer.domElement.addEventListener('webglcontextlost', handleContextLost)
        renderer.domElement.addEventListener('webglcontextrestored', handleContextRestored)

        // Scene
        const scene = new THREE.Scene()
        // Very subtle fog — only noticeable at 200+ meters, never obscures close-up detail
        scene.fog = new THREE.FogExp2(0x0d1117, 0.0003)
        sceneRef.current = scene

        // Environment map for PBR (image-based lighting): a PMREM-prefiltered version
        // of `RoomEnvironment` (a procedural neutral studio). Without this, metals look
        // black and roughness/diffuse don't pick up any ambient gradient. With it,
        // PBR materials look natural even with our few discrete lights.
        try {
            const pmrem = new THREE.PMREMGenerator(renderer)
            const envScene = new RoomEnvironment()
            scene.environment = pmrem.fromScene(envScene, 0.04).texture
            pmrem.dispose()
        } catch (e) {
            console.warn('[Viewport] PMREM environment setup failed:', e)
        }

        // Discrete lights (still useful for shadow direction / sharper highlights;
        // env map provides the soft ambient/diffuse).
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.25)
        scene.add(ambientLight)
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.85)
        dirLight.position.set(10, 20, 10)
        scene.add(dirLight)
        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.30)
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

        // Immersive assistant overlay (animated measurements + evaluation volumes)
        const assistantViz = new AssistantViz()
        scene.add(assistantViz.group)
        assistantVizRef.current = assistantViz

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
        // User interaction wins instantly over any assistant camera flight
        controls.addEventListener('start', () => {
            if (camTweenRef.current != null) {
                cancelAnimationFrame(camTweenRef.current)
                camTweenRef.current = null
            }
        })
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
                uConfHl: { value: -1.0 },
                // 256-slot visibility lookup texture (see vertex shader)
                uSegVisTex: { value: makeSegVisTexture() },
                time: { value: 0 },
                sectionBoxEnabled: { value: false },
                sectionBoxMin: { value: new THREE.Vector3(-100, -100, -100) },
                sectionBoxMax: { value: new THREE.Vector3(100, 100, 100) },
                uSelBoxOn: { value: false },
                uSelBoxInv: { value: new THREE.Matrix4() },
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

            // Advance immersive assistant animations (measurement reveals)
            assistantVizRef.current?.update(now)

            // Potree LOD: update visible nodes based on camera (~10 Hz, not every frame)
            if (potreeLoaderRef.current?.isLoaded && now - lastLodUpdateRef.current > 100) {
                lastLodUpdateRef.current = now
                // Skip Potree's LOD/node-loading work when the cloud is hidden.
                // We can't check octreeGroup.visible: setCloudObjectVisible
                // hides only the potree-node-* children so the mesh groups
                // under octreeGroup stay visible. cloudHiddenRef is the truth.
                if (cloudHiddenRef.current) {
                    if (totalPointsRef.current !== 0) {
                        totalPointsRef.current = 0
                        onPointCount(0)
                    }
                } else {
                    potreeLoaderRef.current.updateVisibility()
                    // Report visible points back to UI
                    const visiblePts = potreeLoaderRef.current.getVisiblePointCount()
                    if (visiblePts !== totalPointsRef.current) {
                        totalPointsRef.current = visiblePts
                        onPointCount(visiblePts)
                    }
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
            if (activeToolRef.current === 'erase') e.preventDefault()   // right-click erases, no browser menu
        }
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') cancelPending()
        }
        // Section box drag handlers
        const onSectionDown = (e: MouseEvent) => handleSectionMouseDown(e)
        const onSectionMove = (e: MouseEvent) => handleSectionMouseMove(e)
        const onSectionUp = () => handleSectionMouseUp()

        // ── Eraser tool (user 2026-08-29): brush a sphere over cloud OR mesh;
        // the backend removes the points from their segment AND clears their
        // mask pixels (deletion survives re-matching), crops the published
        // mesh instantly and debounces a re-fit; the recolored octree arrives
        // via a potree_ready broadcast and reloads in place.
        const eraseSphereGeom = new THREE.SphereGeometry(1, 24, 16)
        const eraseCubeGeom = new THREE.BoxGeometry(2, 2, 2)   // half-extent 1, like the unit sphere
        const eraseCursor: THREE.Mesh = new THREE.Mesh(
            eraseSphereGeom as THREE.BufferGeometry,
            new THREE.MeshBasicMaterial({
                color: 0xff4444, transparent: true, opacity: 0.28,
                depthWrite: false,
            }))
        eraseCursor.visible = false
        eraseCursor.name = 'erase-cursor'
        eraseCursor.userData.shape = 'sphere'
        scene.add(eraseCursor)
        eraseCursorRef.current = eraseCursor

        const eraseRaycast = (event: MouseEvent): THREE.Vector3 | null => {
            const rect = renderer.domElement.getBoundingClientRect()
            const mouse = new THREE.Vector2(
                ((event.clientX - rect.left) / rect.width) * 2 - 1,
                -((event.clientY - rect.top) / rect.height) * 2 + 1)
            const raycaster = raycasterRef.current
            raycaster.params.Points = { threshold: 0.02 }
            raycaster.setFromCamera(mouse, camera)
            const targets: THREE.Object3D[] = []
            if (!cloudHiddenRef.current) {
                if (pointCloudRef.current) targets.push(pointCloudRef.current)
                const octreeGroup = sceneRef.current?.getObjectByName('potree-octree')
                if (octreeGroup) {
                    octreeGroup.children.forEach(c => {
                        if ((c.name || '').startsWith('potree-node-') && c.visible) targets.push(c)
                    })
                }
            }
            if (tsdfGroupRef.current) targets.push(tsdfGroupRef.current)
            if (shapesGroupRef.current) targets.push(shapesGroupRef.current)
            const hits = raycaster.intersectObjects(targets)
            const hit = hits.find(h => h.object !== eraseCursor && hitIsVisible(h))
            return hit ? hit.point.clone() : null
        }
        const onEraseMove = (event: MouseEvent) => {
            if (activeToolRef.current !== 'erase') {
                eraseCursor.visible = false
                setHighlight(null)
                return
            }
            // SHIFT = mark-removal mode: hide the add-cursor, light up the
            // mark under the pointer (the one Shift+right-click will delete)
            if (event.shiftKey) {
                eraseCursor.visible = false
                setHighlight(markUnderCursor(event))
                return
            }
            setHighlight(null)
            if (eraseShapeRef.current === 'box') {
                eraseCursor.visible = false
                return
            }
            const p = eraseRaycast(event)
            if (p) {
                if (eraseCursor.userData.shape !== eraseShapeRef.current) {
                    eraseCursor.geometry = eraseShapeRef.current === 'cube' ? eraseCubeGeom : eraseSphereGeom
                    eraseCursor.userData.shape = eraseShapeRef.current
                }
                eraseCursor.rotation.y = eraseShapeRef.current === 'cube' ? eraseYawRef.current : 0
                eraseCursor.position.copy(p)
                eraseCursor.scale.setScalar(eraseRadiusRef.current)
                eraseCursor.visible = true
            } else {
                eraseCursor.visible = false
            }
        }
        // TWO-PHASE ERASE (user 2026-08-29: "iluminar los puntos y luego poner
        // borrar"): a right-CLICK only MARKS a zone (instant red sphere, no
        // server call); the 'Erase' button in the toolbar sub-panel commits
        // every mark in ONE application (one mask edit, one OBB recompute,
        // one octree rebuild). Navigation stays untouched — right-drag pans.
        const eraseMarksGroup = new THREE.Group()
        eraseMarksGroup.name = 'erase-marks'
        scene.add(eraseMarksGroup)
        type EraseMark = { shape: 'sphere' | 'cube' | 'box'; center?: THREE.Vector3; radius?: number; yawDeg?: number; mesh: THREE.Mesh }
        const eraseMarks: EraseMark[] = []
        const markGeom = new THREE.SphereGeometry(1, 20, 14)
        const markCubeGeom = new THREE.BoxGeometry(2, 2, 2)
        const markMat = new THREE.MeshBasicMaterial({
            color: 0xff3333, transparent: true, opacity: 0.38, depthWrite: false,
        })
        // SHIFT + hover highlight: the mark about to be deleted (user
        // 2026-08-30: with 50 marks, one wrong mark must be removable alone)
        const markHighlightMat = new THREE.MeshBasicMaterial({
            color: 0xffffff, transparent: true, opacity: 0.6, depthWrite: false,
        })
        let highlightedMark: THREE.Mesh | null = null
        const setHighlight = (mesh: THREE.Mesh | null) => {
            if (highlightedMark === mesh) return
            if (highlightedMark) highlightedMark.material = markMat
            highlightedMark = mesh
            if (highlightedMark) highlightedMark.material = markHighlightMat
        }
        const markUnderCursor = (event: MouseEvent): THREE.Mesh | null => {
            if (!eraseMarksGroup.children.length) return null
            const rect = renderer.domElement.getBoundingClientRect()
            const mouse = new THREE.Vector2(
                ((event.clientX - rect.left) / rect.width) * 2 - 1,
                -((event.clientY - rect.top) / rect.height) * 2 + 1)
            const raycaster = raycasterRef.current
            raycaster.setFromCamera(mouse, camera)
            const hits = raycaster.intersectObjects(eraseMarksGroup.children, false)
            return hits.length ? (hits[0].object as THREE.Mesh) : null
        }
        // ── BOX SELECTION (user 2026-08-30: "debe ser una caja, punto"):
        // right-click in box mode drops a box ANYWHERE (surface hit, or 3 m in
        // front of the camera). Left-click selects it → gizmo (Move default;
        // panel buttons or G/R/S; Esc done; Del removes). While selected,
        // every point INSIDE lights up golden (shader highlight).
        let boxTc: TransformControls | null = null
        let selBox: THREE.Mesh | null = null
        const _selBoxInv = new THREE.Matrix4()
        const _updateSelBoxUniform = () => {
            const m = materialRef.current
            if (!m) return
            if (selBox) {
                selBox.updateMatrixWorld(true)
                _selBoxInv.copy(selBox.matrixWorld).invert()
                ;(m.uniforms.uSelBoxInv.value as THREE.Matrix4).copy(_selBoxInv)
                m.uniforms.uSelBoxOn.value = true
            } else {
                m.uniforms.uSelBoxOn.value = false
            }
        }
        const boxDeselect = () => {
            if (boxTc) {
                scene.remove(boxTc.getHelper())
                boxTc.detach()
                boxTc.dispose()
                boxTc = null
            }
            selBox = null
            _updateSelBoxUniform()
            onEraseBoxSelectedRef.current?.(false)
            const ctrl = controlsRef.current
            if (ctrl) ctrl.enabled = true
        }
        const boxSelect = (mesh: THREE.Mesh) => {
            boxDeselect()
            selBox = mesh
            const tc = new TransformControls(camera, renderer.domElement)
            tc.setMode('translate')
            tc.setSize(0.7)
            tc.attach(mesh)
            scene.add(tc.getHelper())
            tc.addEventListener('dragging-changed', (e) => {
                const ctrl = controlsRef.current
                if (ctrl) ctrl.enabled = !(e as unknown as { value?: boolean }).value
            })
            tc.addEventListener('objectChange', _updateSelBoxUniform)
            boxTc = tc
            _updateSelBoxUniform()
            onEraseBoxSelectedRef.current?.(true)
            if (onStatusMessage) onStatusMessage('🔳 box selected — Move/Rotate/Stretch (panel or G/R/S) · Esc done · Del remove')
        }
        eraseBoxApiRef.current = {
            setMode: (m: 'translate' | 'rotate' | 'scale') => boxTc?.setMode(m),
            remove: () => {
                if (!selBox) return
                const mesh = selBox
                boxDeselect()
                const idx = eraseMarks.findIndex(mk => mk.mesh === mesh)
                if (idx >= 0) eraseMarks.splice(idx, 1)
                eraseMarksGroup.remove(mesh)
                onEraseMarksChangedRef.current?.(eraseMarks.length)
            },
        }
        const onEraseKey = (e: KeyboardEvent) => {
            if (activeToolRef.current !== 'erase') return
            if (e.key === 'Escape' && selBox) { boxDeselect(); return }
            if (!selBox || !boxTc) return
            const k = e.key.toLowerCase()
            if (k === 'g') boxTc.setMode('translate')
            else if (k === 'r') boxTc.setMode('rotate')
            else if (k === 's') boxTc.setMode('scale')
            else if (e.key === 'Delete') eraseBoxApiRef.current?.remove()
        }
        const eraseLeftDown = { x: 0, y: 0, active: false }
        const onEraseLeftDown = (e: MouseEvent) => {
            if (activeToolRef.current !== 'erase' || e.button !== 0) return
            eraseLeftDown.x = e.clientX
            eraseLeftDown.y = e.clientY
            eraseLeftDown.active = true
        }
        const onEraseLeftUp = (e: MouseEvent) => {
            if (activeToolRef.current !== 'erase' || e.button !== 0) return
            if (!eraseLeftDown.active) return
            eraseLeftDown.active = false
            if (Math.hypot(e.clientX - eraseLeftDown.x, e.clientY - eraseLeftDown.y) > 4) return
            const rect = renderer.domElement.getBoundingClientRect()
            const mouse = new THREE.Vector2(
                ((e.clientX - rect.left) / rect.width) * 2 - 1,
                -((e.clientY - rect.top) / rect.height) * 2 + 1)
            const raycaster = raycasterRef.current
            raycaster.setFromCamera(mouse, camera)
            const hits = raycaster.intersectObjects(eraseMarksGroup.children, false)
            const hit = hits.find(h => (h.object as THREE.Mesh).userData.gizmoBox)
            if (hit) boxSelect(hit.object as THREE.Mesh)
            else if (selBox) boxDeselect()
        }

        const eraseClearMarks = () => {
            setHighlight(null)
            boxDeselect()
            eraseMarks.length = 0
            while (eraseMarksGroup.children.length) {
                eraseMarksGroup.remove(eraseMarksGroup.children[0])
            }
            onEraseMarksChangedRef.current?.(0)
        }
        // commit: no target → DELETE; target id → REASSIGN into that segment;
        // newLabel → CREATE a new segment from the zones (user 2026-08-29)
        // shared POST + transaction-trace reporting for every brush apply
        // (zones commit AND confidence filter)
        const postErase = async (payload: Record<string, unknown>, newLabel?: string) => {
                const res = await fetch('/api/segmentation/erase', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                })
                const data = await res.json()
                if (onStatusMessage) {
                    // Transaction trace (user 2026-08-31): show EXACTLY what the
                    // backend did — target, per-source breakdown, protected
                    // hidden points, balance and file verification.
                    const led = data.ledger
                    const nDel = data.total_removed || 0
                    const nRe = data.reassigned || 0
                    if (led) {
                        const parts: string[] = []
                        const from = Object.entries(led.moved_from || {})
                            .map(([k, n]) => `${k}: ${(n as number).toLocaleString()}`)
                        if (led.unsegmented_taken) from.push(`unsegmented: ${led.unsegmented_taken.toLocaleString()}`)
                        if (led.mode === 'reassign' && led.target) {
                            parts.push(`🖌 ${(led.total_moved || 0).toLocaleString()} pts → ${led.target.label}_${led.target.instance_id}`)
                        } else if (led.mode === 'delete') {
                            parts.push(`🧽 ${(led.total_moved || 0).toLocaleString()} pts erased`)
                        }
                        if (from.length) parts.push(`from ${from.join(', ')}`)
                        if (led.balance) parts.push(`balance ${led.balance.consistent ? '✓' : '✗ MISMATCH'}`)
                        if (led.files_verified !== null && led.files_verified !== undefined)
                            parts.push(`files ${led.files_verified ? '✓' : '✗ MISMATCH'}`)
                        if (led.exclusive !== undefined)
                            parts.push(led.exclusive ? 'exclusive ✓'
                                : `✗ ${(led.overlap_points || 0).toLocaleString()} pts owned by >1 segment`)
                        if (led.deleted_points) parts.push(`🗑 ${led.deleted_points.toLocaleString()} unsegmented pts DELETED from the cloud (irreversible)`)
                        const prot = Object.entries(led.protected_hidden || {})
                        if (prot.length) {
                            const nProt = prot.reduce((a, [, n]) => a + (n as number), 0)
                            parts.push(`⚠ ${nProt.toLocaleString()} pts in HIDDEN segments untouched (${prot.map(([k]) => k).join(', ')})`)
                        }
                        if (!led.total_moved && !prot.length) parts.push('no applicable points in the zones')
                        onStatusMessage(parts.join(' · '))
                    } else if (nRe) onStatusMessage(`🖌 ${nRe.toLocaleString()} points ${newLabel ? `→ new segment "${newLabel}"` : 'reassigned'} — recoloring...`)
                    else if (nDel) onStatusMessage(`🧽 ${nDel.toLocaleString()} points erased from ${Object.keys(data.touched || {}).length} object(s) — recoloring...`)
                    else onStatusMessage('🖌 the marked zones had no applicable points')
                }
                onEraseLedgerRef.current?.(data.ledger ?? null)
        }
        const eraseCommit = async (target?: number | null, newLabel?: string, onlyInstances?: number[], includeUnsegmented?: boolean) => {
            const sid = activeSessionRef.current
            if (!sid || !eraseMarks.length) return
            try {
                if (onStatusMessage) onStatusMessage(`🖌 applying ${eraseMarks.length} zone(s)...`)
                const payload: Record<string, unknown> = {
                    session_id: sid,
                    spheres: eraseMarks.map(m => {
                        if (m.shape === 'box') {
                            m.mesh.updateMatrixWorld(true)
                            return { shape: 'box', matrix: m.mesh.matrixWorld.toArray() }
                        }
                        return {
                            center: [m.center!.x, m.center!.y, m.center!.z],
                            radius: m.radius,
                            shape: m.shape,
                            yaw_deg: m.yawDeg,
                        }
                    }),
                }
                // SAFETY: hidden segments cannot lose points
                if (onlyInstances) payload.only_instances = onlyInstances
                // unsegmented points join a reassign only when their toggle is ON
                if (includeUnsegmented !== undefined) payload.include_unsegmented = includeUnsegmented
                if (newLabel) {
                    payload.reassign_to = 'new'
                    payload.new_label = newLabel
                } else if (target !== undefined && target !== null) {
                    payload.reassign_to = target
                }
                await postErase(payload, newLabel)
                eraseClearMarks()
            } catch {
                if (onStatusMessage) onStatusMessage('🖌 apply failed')
            }
        }
        // confidence filter (user 2026-08-31): points below the threshold go
        // to unsegmented — no zones needed, same safety and same ledger
        const eraseConfApply = async (thr: number, onlyInstances?: number[], includeUnsegmented?: boolean) => {
            const sid = activeSessionRef.current
            if (!sid) return
            try {
                if (onStatusMessage) onStatusMessage(`🎚 applying confidence filter < ${(thr * 100).toFixed(0)}%...`)
                await postErase({
                    session_id: sid,
                    spheres: [],
                    conf_below: thr,
                    only_instances: onlyInstances,
                    // Unsegmented toggle ON → its low-confidence points are
                    // PHYSICALLY DELETED from the cloud (user 2026-08-31)
                    include_unsegmented: includeUnsegmented ?? false,
                })
            } catch {
                if (onStatusMessage) onStatusMessage('🎚 confidence filter failed')
            }
        }
        eraseApiRef.current = { commit: eraseCommit, clear: eraseClearMarks, confApply: eraseConfApply }

        const eraseRightDown = { x: 0, y: 0, active: false, shift: false }
        const onEraseMouseDown = (event: MouseEvent) => {
            if (activeToolRef.current !== 'erase' || event.button !== 2) return
            eraseRightDown.x = event.clientX
            eraseRightDown.y = event.clientY
            eraseRightDown.active = true
            eraseRightDown.shift = event.shiftKey
        }
        const onEraseMouseUp = (event: MouseEvent) => {
            if (activeToolRef.current !== 'erase' || event.button !== 2) return
            if (!eraseRightDown.active) return
            eraseRightDown.active = false
            const moved = Math.hypot(event.clientX - eraseRightDown.x,
                                     event.clientY - eraseRightDown.y)
            if (moved > 6) return   // that was a pan, not a mark
            if (eraseRightDown.shift || event.shiftKey) {
                // SHIFT + right-click: remove ONLY the mark under the cursor
                const mesh = markUnderCursor(event)
                if (!mesh) return
                setHighlight(null)
                if (mesh === selBox) boxDeselect()
                const idx = eraseMarks.findIndex(m => m.mesh === mesh)
                if (idx >= 0) eraseMarks.splice(idx, 1)
                eraseMarksGroup.remove(mesh)
                onEraseMarksChangedRef.current?.(eraseMarks.length)
                if (onStatusMessage) onStatusMessage(`🖌 mark removed — ${eraseMarks.length} zone(s) left`)
                return
            }
            if (eraseShapeRef.current === 'box') {
                // drop a selection box ANYWHERE: at the surface hit, or 3 m in
                // front of the camera when the click hits nothing
                const hitP = eraseRaycast(event)
                const fwd = camera.getWorldDirection(new THREE.Vector3())
                const pos = hitP ?? camera.getWorldPosition(new THREE.Vector3()).addScaledVector(fwd, 3)
                const box = new THREE.Mesh(markCubeGeom, markMat)
                box.userData.gizmoBox = true
                box.position.copy(pos)
                box.scale.set(0.75, 0.75, 0.75)   // 1.5 m default box
                eraseMarksGroup.add(box)
                eraseMarks.push({ shape: 'box', mesh: box })
                onEraseMarksChangedRef.current?.(eraseMarks.length)
                boxSelect(box)
                return
            }
            const p = eraseRaycast(event)
            if (!p) return
            const r = eraseRadiusRef.current
            const shape = eraseShapeRef.current
            const mark = new THREE.Mesh(shape === 'cube' ? markCubeGeom : markGeom, markMat)
            mark.position.copy(p)
            mark.scale.setScalar(r)
            const yawDeg = shape === 'cube' ? (eraseYawRef.current * 180) / Math.PI : 0
            if (shape === 'cube') mark.rotation.y = eraseYawRef.current
            eraseMarksGroup.add(mark)
            eraseMarks.push({ shape, center: p.clone(), radius: r, yawDeg, mesh: mark })
            onEraseMarksChangedRef.current?.(eraseMarks.length)
            if (onStatusMessage) onStatusMessage(`🖌 ${eraseMarks.length} zone(s) marked — press Erase or Assign to apply`)
        }

        renderer.domElement.addEventListener('click', onCanvasClick)
        renderer.domElement.addEventListener('contextmenu', onContextMenu)
        renderer.domElement.addEventListener('mousedown', onSectionDown)
        renderer.domElement.addEventListener('mousemove', onSectionMove)
        renderer.domElement.addEventListener('mousemove', handleMeasureHover)
        renderer.domElement.addEventListener('mouseup', onSectionUp)
        renderer.domElement.addEventListener('mousemove', onEraseMove)
        renderer.domElement.addEventListener('mousedown', onEraseMouseDown)
        renderer.domElement.addEventListener('mouseup', onEraseMouseUp)
        renderer.domElement.addEventListener('mousedown', onEraseLeftDown)
        renderer.domElement.addEventListener('mouseup', onEraseLeftUp)
        window.addEventListener('keydown', onEraseKey)
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
            renderer.domElement.removeEventListener('mousemove', onEraseMove)
            renderer.domElement.removeEventListener('mousedown', onEraseMouseDown)
            renderer.domElement.removeEventListener('mouseup', onEraseMouseUp)
            renderer.domElement.removeEventListener('mousedown', onEraseLeftDown)
            renderer.domElement.removeEventListener('mouseup', onEraseLeftUp)
            window.removeEventListener('keydown', onEraseKey)
            boxDeselect()
            eraseBoxApiRef.current = null
            scene.remove(eraseCursor)
            eraseSphereGeom.dispose()
            eraseCubeGeom.dispose()
            ;(eraseCursor.material as THREE.Material).dispose()
            eraseCursorRef.current = null
            scene.remove(eraseMarksGroup)
            markGeom.dispose()
            markCubeGeom.dispose()
            markMat.dispose()
            markHighlightMat.dispose()
            eraseApiRef.current = null
            renderer.domElement.removeEventListener('webglcontextlost', handleContextLost)
            renderer.domElement.removeEventListener('webglcontextrestored', handleContextRestored)
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

    // Update LOD point budget when prop changes → re-evaluate visible nodes
    useEffect(() => {
        const loader = potreeLoaderRef.current
        if (loader) {
            loader.setPointBudget(pointBudget)
            loader.updateVisibility()
        }
    }, [pointBudget])

    // Update confidence threshold when prop changes
    useEffect(() => {
        if (materialRef.current) {
            materialRef.current.uniforms.uConfidenceThreshold.value = confidenceThreshold
        }
        // Confidence-slider "hide cloud" — same per-child path as
        // setCloudObjectVisible so the mesh groups under octreeGroup don't
        // get dragged out of view.
        const hideCloud = confidenceThreshold >= 1.0
        cloudHiddenRef.current = hideCloud
        if (pointCloudRef.current) {
            pointCloudRef.current.visible = !hideCloud
        }
        const potreeGroup = sceneRef.current?.getObjectByName('potree-octree')
        if (potreeGroup) {
            potreeGroup.children.forEach(child => {
                if ((child.name || '').startsWith('potree-node-')) {
                    child.visible = !hideCloud
                }
            })
        }
        // re-showing must not resurrect nodes of hidden segments
        if (!hideCloud) {
            const hidden = new Set<number>()
            segVisRef.current.forEach((vis, id) => { if (!vis) hidden.add(id) })
            potreeLoaderRef.current?.setClassVisibility(hidden)
        }
    }, [confidenceThreshold])

    // Track activeSession in a ref for the WS onopen handler
    const activeSessionRef = useRef(activeSession)
    // Guards the two AUTOMATIC load_session triggers (WS onopen + the
    // activeSession effect) against firing twice for the same session on one
    // connection — that double-fire made the backend reload the cloud AND
    // re-download the (large) TSDF .glb twice. Holds the session we've already
    // auto-loaded on the current socket; reset to null on disconnect (so a
    // reconnect re-requests) and whenever the active session changes. Explicit
    // reloads (sábana toggle, pipeline-done) bypass this via sendCommand*.
    const sentLoadForSessionRef = useRef<string | null>(null)
    useEffect(() => {
        activeSessionRef.current = activeSession
        sessionFramedRef.current = null  // reset so new session gets framed
        sentLoadForSessionRef.current = null  // new session → allow one auto-load
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
                if (activeSessionRef.current && !pipelineRunningRef.current
                    && sentLoadForSessionRef.current !== activeSessionRef.current) {
                    sentLoadForSessionRef.current = activeSessionRef.current
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
                // Socket is gone — let onopen re-request the active session after
                // reconnect (the backend lost our load on the dead socket).
                sentLoadForSessionRef.current = null
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
        // Already auto-loaded this session on this socket (e.g. onopen got there
        // first) → don't fire a second load that reloads the cloud + re-fetches
        // the TSDF .glb.
        if (sentLoadForSessionRef.current === activeSession) return
        sentLoadForSessionRef.current = activeSession

        // Clear existing scene before loading
        clearScene()

        ws.send(JSON.stringify({
            type: 'load_session',
            session_id: activeSession,
        }))
    }, [activeSession])

    // Clear geometry and OBBs
    const clearScene = useCallback(() => {
        // Reset stale view-state refs so a fresh load is never blocked:
        //  - cloudHiddenRef: if it stuck `true` (e.g. all segments hidden), the
        //    LOD loop would skip loading the cloud entirely on the next load.
        //  - preserveCameraRef: set-once and otherwise never cleared → a later
        //    full load would skip frameCloud and restore a stale camera, leaving
        //    the frustum pointing at nothing (cloud "loads" but shows no points).
        // F5 reset these (fresh module state); unload→reload did not — that was
        // exactly the "reload shows only camera poses, no cloud" bug.
        cloudHiddenRef.current = false
        preserveCameraRef.current = null
        sessionFramedRef.current = null
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
        // Clear OBBs (deep: containers hold LineSegments + label Sprites with
        // CanvasTextures — shallow geometry-only disposal leaked them all)
        const group = obbGroupRef.current
        if (group) {
            while (group.children.length > 0) {
                const child = group.children[0]
                group.remove(child)
                child.traverse(obj => {
                    const mesh = obj as THREE.Mesh
                    if (mesh.geometry) mesh.geometry.dispose()
                    const mat = mesh.material as THREE.Material | THREE.Material[] | undefined
                    if (mat) {
                        for (const m of Array.isArray(mat) ? mat : [mat]) {
                            (m as unknown as { map?: { dispose?: () => void } }).map?.dispose?.()
                            m.dispose()
                        }
                    }
                })
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
        // Clear tooltips
        setCamTooltip(null)
        setBimTooltip(null)
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
        if (msg.type === 'tsdf_ready') {
            // freshly-published meshes (poisson / pgsr stage finished) — load
            // them into the scene and let App refresh the panel list
            // (user 2026-08-31: meshes must appear the moment they finish)
            const sid = msg.session_id as string
            if (sid && sid === activeSessionRef.current) {
                const parent = potreeLoaderRef.current?.getOctreeGroup() || sceneRef.current
                if (parent) loadTsdfIntoGroup(sid, parent)
                onTsdfReadyRef.current?.(sid, msg.stage as string)
                if (onStatusMessage) onStatusMessage(
                    `🧩 ${msg.count} ${msg.stage} mesh(es) ready — list updated`)
            }
            return
        }
        if (msg.type === 'erase_verified') {
            // backend sampled the rebuilt octree against classification.npy —
            // the last leg of the brush-transaction trace (user 2026-08-31)
            if (onStatusMessage) {
                const pct = ((msg.agreement as number) * 100).toFixed(2)
                onStatusMessage(msg.ok
                    ? `octree verified ✓ ${msg.matched}/${msg.checked} (${pct}%)`
                    : `⚠ OCTREE MISMATCH: ${msg.matched}/${msg.checked} (${pct}%) ${msg.error || ''} — viewer may show stale classes`)
            }
            return
        }
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

            // Keep loaded mesh groups ALIVE across the octree reload: they are
            // parented under the octreeGroup (floor-transform inheritance), and
            // the old loader's dispose() used to drop them undisposed — GPU
            // memory leaked and meshes vanished on every erase commit/refresh
            // (user 2026-08-30: "se pone pesada con el uso").
            const survivingGroups: THREE.Object3D[] = []
            for (const ref of [tsdfGroupRef, shapesGroupRef, reconSceneGroupRef]) {
                const g = ref.current
                if (g) {
                    g.removeFromParent()
                    survivingGroups.push(g)
                }
            }

            // Dispose previous loader
            if (potreeLoaderRef.current) {
                potreeLoaderRef.current.dispose()
            }

            const loader = new PotreeOctreeLoader(scene, camera, mat, pointBudget)
            potreeLoaderRef.current = loader
            if (survivingGroups.length) {
                const newOctree = scene.getObjectByName('potree-octree')
                for (const g of survivingGroups) (newOctree ?? scene).add(g)
            }
            // restore this session's placed library objects
            if (msg.session_id) reloadSceneObjectsRef.current(msg.session_id as string)
            // carry the user's hidden segments into the fresh loader
            {
                const hidden = new Set<number>()
                segVisRef.current.forEach((vis, id) => { if (!vis) hidden.add(id) })
                if (hidden.size) loader.setClassVisibility(hidden)
            }

            loader.load(url).then((loadedPts) => {
                // Guard: if this loader was replaced (e.g. by sábana), skip
                if (potreeLoaderRef.current !== loader) return
                if (onStatusMessage) onStatusMessage(`LOD octree loaded — ${pts?.toLocaleString()} total points`)
                onPointCount(loadedPts)
                // undefined = the sender didn't say (older erase-rebuild
                // broadcasts) — never DROP a known-true confidence over that
                if (onHasConfidence && serverHasConfidence !== undefined) onHasConfidence(!!serverHasConfidence)

                // resync OBBs + side panel: an erase/refresh may have changed
                // the instances behind this reload (user 2026-08-30: bbox
                // stayed stale after erasing)
                const sid = activeSessionRef.current
                if (sid) {
                    fetch(`/api/sessions/${sid}/segmentation`)
                        .then(r => (r.ok ? r.json() : null))
                        .then(data => {
                            if (data && Array.isArray(data.instances)) {
                                renderOBBsRef.current?.(data.instances)
                            }
                        })
                        .catch(() => { })
                }

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
                    // Restore saved camera state if preserving (one-shot: clear
                    // after use so a later full load re-frames instead of
                    // restoring a stale camera).
                    if (preserveCameraRef.current && cameraRef.current && controlsRef.current) {
                        cameraRef.current.position.copy(preserveCameraRef.current.pos)
                        controlsRef.current.target.copy(preserveCameraRef.current.target)
                        controlsRef.current.update()
                        preserveCameraRef.current = null
                    }
                    // Adapt grid to combined extents
                    if (sceneRef.current) adaptGrid(cloudBBoxRef.current, sceneRef.current, gridRef, showGridRef.current)
                }

                // Auto-load reconstructed object meshes (ShapeR .glb) into the
                // octreeGroup so floor_transform applies automatically.
                const sessionIdForShapes = msg.session_id as string | undefined
                const octreeGroupForShapes = loader.getOctreeGroup()
                if (sessionIdForShapes && octreeGroupForShapes) {
                    loadShapesIntoGroup(sessionIdForShapes, octreeGroupForShapes)
                    loadTsdfIntoGroup(sessionIdForShapes, octreeGroupForShapes)
                    const preloadedScene = msg.scene as { elements?: Array<any> } | undefined
                    loadReconSceneIntoGroup(sessionIdForShapes, octreeGroupForShapes, preloadedScene)
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
            const loader = new PotreeOctreeLoader(scene, camera, mat, pointBudget, -1)
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
    // Deep-dispose a three.js subtree: geometry + material(s) + their texture
    // maps. The OBB containers are Groups whose children (edge LineSegments,
    // label Sprites with CanvasTextures) were being removed WITHOUT disposal —
    // every segments refresh leaked GPU buffers/textures, which is why the
    // viewer slowed down over time and got WORSE after deleting elements
    // (each delete triggers a full OBB rebuild → another leaked generation).
    const disposeDeep = useCallback((root: THREE.Object3D) => {
        root.traverse(obj => {
            const mesh = obj as THREE.Mesh
            if (mesh.geometry) mesh.geometry.dispose()
            const mat = mesh.material as THREE.Material | THREE.Material[] | undefined
            if (mat) {
                const mats = Array.isArray(mat) ? mat : [mat]
                for (const m of mats) {
                    const anyM = m as unknown as Record<string, { dispose?: () => void } | undefined>
                    for (const texKey of ['map', 'alphaMap', 'aoMap', 'emissiveMap', 'normalMap', 'roughnessMap', 'metalnessMap']) {
                        anyM[texKey]?.dispose?.()
                    }
                    m.dispose()
                }
            }
        })
    }, [])

    const renderOBBs = useCallback((instances: Array<Record<string, unknown>>) => {
        const group = obbGroupRef.current
        if (!group) return

        // Preserve the user's choices across re-renders (user 2026-08-30:
        // recomputing a bbox must not bring every hidden box/segment back)
        const prevObbVis = new Map<string, boolean>()
        obbMapRef.current.forEach((c, k) => prevObbVis.set(k, c.visible))

        // Clear old OBBs (deep: containers hold LineSegments + label Sprites
        // with CanvasTextures — see disposeDeep)
        while (group.children.length > 0) {
            const child = group.children[0]
            group.remove(child)
            disposeDeep(child)
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
                visible: segVisRef.current.get(instId) ?? true,
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

            obbContainer.visible = prevObbVis.get(globalKey) ?? true
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
    useEffect(() => { renderOBBsRef.current = renderOBBs }, [renderOBBs])

    return (
        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
            <div ref={containerRef} className="viewport-canvas" style={{ width: '100%', height: '100%' }} />
            {contextLost && (
                <div style={{
                    position: 'absolute', inset: 0, zIndex: 500,
                    display: 'flex', flexDirection: 'column', alignItems: 'center',
                    justifyContent: 'center', gap: 12, textAlign: 'center',
                    background: 'rgba(13, 17, 23, 0.94)', color: '#e6edf3', padding: 24,
                }}>
                    <div style={{ fontSize: 34 }}>🖥️⚠️</div>
                    <div style={{ fontSize: 16, fontWeight: 700 }}>WebGL context lost</div>
                    <div style={{ fontSize: 13, color: '#8b949e', maxWidth: 460 }}>
                        The GPU ran out of resources rendering this scene. If it does
                        not recover automatically, lower the Detail (point budget)
                        slider and reload.
                    </div>
                    <button
                        onClick={() => window.location.reload()}
                        style={{
                            padding: '8px 20px', border: 'none', borderRadius: 8,
                            background: 'var(--accent, #2f81f7)', color: '#fff',
                            fontSize: 13, fontWeight: 600, cursor: 'pointer',
                        }}
                    >↻ Reload</button>
                </div>
            )}
            {activeTool === 'align' && (
                <div style={{
                    position: 'absolute', top: 12, right: 12, background: 'var(--glass-bg)',
                    borderRadius: 10, padding: '14px 18px', color: '#fff', fontSize: 13,
                    display: 'flex', flexDirection: 'column', gap: 10, minWidth: 180,
                    border: '1px solid var(--glass-border)', backdropFilter: 'blur(8px)',
                    zIndex: 100
                }}>
                    <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 2 }}>⛶ Align Cloud</div>
                    <div style={{ display: 'flex', gap: 6 }}>
                        <button
                            style={{ flex: 1, padding: '6px 0', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 12, fontWeight: 600, background: alignMode === 'rotate' ? 'var(--accent)' : 'var(--bg-active)', color: '#fff' }}
                            onClick={() => setAlignMode('rotate')}
                        >🔄 Rotate</button>
                        <button
                            style={{ flex: 1, padding: '6px 0', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 12, fontWeight: 600, background: alignMode === 'translate' ? 'var(--accent-2)' : 'var(--bg-active)', color: '#fff' }}
                            onClick={() => setAlignMode('translate')}
                        >↔️ Move</button>
                    </div>
                    <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: '2px 0' }} />
                    <button
                        style={{ padding: '8px 0', border: 'none', borderRadius: 6, cursor: alignDirty ? 'pointer' : 'not-allowed', fontSize: 13, fontWeight: 700, background: alignDirty ? 'var(--success)' : 'var(--bg-active)', color: '#fff', opacity: alignDirty ? 1 : 0.5 }}
                        onClick={saveAlignment}
                        disabled={!alignDirty}
                    >💾 Save Alignment</button>
                </div>
            )}
            {/* Evaluation-volume gizmo toolbar (click a volume to select it) */}
            {selVolume != null && (
                <div style={{
                    position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)',
                    background: 'var(--glass-bg)', borderRadius: 10, padding: '8px 12px',
                    color: '#fff', fontSize: 12, display: 'flex', gap: 6, alignItems: 'center',
                    border: '1px solid var(--glass-border)', backdropFilter: 'blur(8px)', zIndex: 100,
                }}>
                    <span style={{ fontWeight: 700, color: '#c4b5ff', marginRight: 4 }}>⬚ Volume #{selVolume}</span>
                    {([['translate', '↔ Move'], ['rotate', '🔄 Rotate'], ['scale', '⤢ Resize']] as const).map(([m, lbl]) => (
                        <button key={m}
                            style={{ padding: '5px 10px', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 12, fontWeight: 600, background: volMode === m ? 'var(--accent-2)' : 'var(--bg-active)', color: '#fff' }}
                            onClick={() => setVolMode(m)}>{lbl}</button>
                    ))}
                    <button
                        style={{ padding: '5px 10px', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 12, fontWeight: 600, background: volSolid ? 'var(--accent)' : 'var(--bg-active)', color: '#fff' }}
                        onClick={() => {
                            const s = !volSolid
                            setVolSolid(s)
                            assistantVizRef.current?.setVolumeSolid(selVolume, s)
                        }}>◼ Solid</button>
                    <button
                        style={{ padding: '5px 10px', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 12, fontWeight: 600, background: 'var(--danger, #f85149)', color: '#fff' }}
                        onClick={() => {
                            const vid = selVolume
                            setSelVolume(null)
                            assistantVizRef.current?.removeVolume(vid)
                            onVolumeDeletedRef.current?.(vid)
                        }}>🗑</button>
                    <button
                        style={{ padding: '5px 8px', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 12, background: 'var(--bg-active)', color: '#fff' }}
                        onClick={() => setSelVolume(null)}>✕</button>
                </div>
            )}
            {/* Camera pose hover tooltip */}
            {camTooltip && (
                <div style={{
                    position: 'absolute',
                    left: camTooltip.x + 16,
                    top: camTooltip.y - 80,
                    background: 'var(--glass-bg)',
                    borderRadius: 8,
                    padding: 8,
                    color: '#fff',
                    fontSize: 11,
                    fontWeight: 600,
                    zIndex: 200,
                    pointerEvents: 'none',
                    border: '1px solid var(--glass-border)',
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
                    background: 'var(--glass-bg)',
                    borderRadius: 6,
                    padding: '5px 10px',
                    color: '#fff',
                    fontSize: 11,
                    zIndex: 200,
                    pointerEvents: 'none',
                    border: '1px solid rgba(79, 209, 255, 0.3)',
                    backdropFilter: 'blur(6px)',
                    maxWidth: 360,
                    wordBreak: 'break-all' as const,
                }}>
                    <span style={{ color: 'var(--accent-2)', fontWeight: 600 }}>{bimTooltip.type.replace('Ifc', '')}</span>
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
                    background: 'var(--glass-bg)',
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
                        <span style={{ color: 'var(--accent-2)', fontWeight: 600 }}>{bimCtxMenu.type.replace('Ifc', '')}</span>
                        {bimCtxMenu.name && !bimCtxMenu.name.startsWith('Element_') && (
                            <span style={{ marginLeft: 6 }}>{bimCtxMenu.name}</span>
                        )}
                    </div>
                    {/* Transparency slider */}
                    <div style={{ padding: '8px 12px', borderBottom: '1px solid var(--border-light)' }}>
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
                                style={{ flex: 1, accentColor: 'var(--accent-2)' }}
                            />
                            <span style={{ fontSize: 10, opacity: 0.5, minWidth: 28, textAlign: 'right' }}>
                                {Math.round((1 - bimCtxMenu.opacity) * 100)}%
                            </span>
                        </div>
                    </div>
                    {/* Hide */}
                    <div
                        style={{ padding: '8px 12px', cursor: 'pointer', borderBottom: '1px solid var(--border-light)' }}
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
                        style={{ padding: '8px 12px', cursor: 'pointer', borderBottom: '1px solid var(--border-light)' }}
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
