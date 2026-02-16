/**
 * STAC Build — 3D Viewport Component
 * Three.js-based point cloud renderer
 * Hernán Barreto — Ingerop IN3
 */
import { useRef, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

interface ViewportProps {
    pointSize: number
    activeSession: string | null
    onPointCount: (count: number) => void
    onFps: (fps: number) => void
    onStatusMessage?: (msg: string) => void
    onSegments?: (segments: SegmentInstance[]) => void
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
}

// Vertex shader — matches FusionRenderer.js point size formula
const vertexShader = `
  attribute float classId;
  varying float vClassId;
  varying vec3 vColor;
  uniform float pointSize;

  void main() {
    vClassId = classId;
    vColor = color;
    
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    
    // Same formula as standalone viewer: pointScale / depth
    float depth = -mvPosition.z;
    float size = pointSize / depth;
    gl_PointSize = clamp(size, 1.0, 5.0);
  }
`

// Fragment shader — circular points with EDL-like shading
const fragmentShader = `
  varying float vClassId;
  varying vec3 vColor;
  uniform float highlightIntensity;

  void main() {
    vec2 centered = gl_PointCoord - 0.5;
    float dist = length(centered);
    if (dist > 0.5) discard;
    
    float alpha = 1.0 - smoothstep(0.35, 0.5, dist);
    
    vec3 finalColor = vColor * 0.85;
    
    gl_FragColor = vec4(finalColor, alpha);
  }
`

const Viewport = forwardRef<ViewportHandle, ViewportProps>(function Viewport(
    { pointSize, activeSession, onPointCount, onFps, onStatusMessage, onSegments },
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

    // Expose sendCommand + toggleOBB to parent via ref
    useImperativeHandle(ref, () => ({
        sendCommand: (cmd: Record<string, unknown>) => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify(cmd))
            }
        },
        toggleOBB: (key: string, visible: boolean) => {
            const mesh = obbMapRef.current.get(key)
            if (mesh) mesh.visible = visible
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
        controls.minDistance = 0.1
        controls.maxDistance = 500
        controls.maxPolarAngle = Math.PI
        controlsRef.current = controls

        // Shader material for point cloud
        const material = new THREE.ShaderMaterial({
            vertexShader,
            fragmentShader,
            uniforms: {
                pointSize: { value: pointSize },
                highlightIntensity: { value: 0.5 },
                time: { value: 0 },
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

        // Cleanup
        return () => {
            cancelAnimationFrame(animFrameRef.current)
            resizeObserver.disconnect()
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

    // Connect WebSocket once on mount — stays alive across session changes
    useEffect(() => {
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
        }

        return () => {
            ws.close()
        }
    }, []) // Only once on mount

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

    // Handle JSON messages from server (segmentation, status, etc)
    const handleJsonMessage = useCallback((msg: Record<string, unknown>) => {
        // v2.0 format: { instances: [{label, instance_id, color, total_points, obb, ...}] }
        if (Array.isArray(msg.instances)) {
            renderOBBs(msg.instances as Array<Record<string, unknown>>)
        }
        if (msg.type === 'status' || msg.type === 'progress') {
            const text = (msg.message || msg.status || '') as string
            if (text && onStatusMessage) onStatusMessage(text)
        }
    }, [onStatusMessage, onSegments])

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

        console.log(`[Viewport] Rendered ${group.children.length} OBBs`)

        // Notify parent with segment list
        if (onSegments) onSegments(segmentList)
    }, [onSegments])

    return <div ref={containerRef} className="viewport-canvas" />
})

export default Viewport
