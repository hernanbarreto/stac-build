/**
 * STAC-BUILD: FusionRenderer
 * Three.js-based point cloud renderer with splatting shaders
 * ES Modules compatible
 * FIX: Added accumulation logic to prevent overwriting
 */

import * as THREE from './libs/three.module.js';
import { OrbitControls } from './libs/OrbitControls.js';

// Vertex Shader - depth-based point size for splatting effect
const vertexShader = `
attribute float classId;
attribute vec3 color;

varying float vClassId;
varying vec3 vColor;

uniform float pointScale;
uniform float minPointSize;
uniform float maxPointSize;

void main() {
    vClassId = classId;
    vColor = color;
    
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    
    // Depth-based point size (splatting effect)
    float depth = -mvPosition.z;
    float size = pointScale / depth;
    gl_PointSize = clamp(size, minPointSize, maxPointSize);
}
`;

// Fragment Shader - circular points with class-based coloring
const fragmentShader = `
varying float vClassId;
varying vec3 vColor;

uniform float time;
uniform float highlightIntensity;
uniform float backgroundBrightness;

void main() {
    // Circular point shape
    vec2 centered = gl_PointCoord - 0.5;
    float dist = length(centered);
    if (dist > 0.5) discard;
    
    // Soft edge
    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    
    vec3 finalColor;
    
    if (vClassId > 0.0) {
        // Highlighted/segmented objects - pulsing orange/yellow
        float pulse = 0.7 + 0.3 * sin(time * 5.0 + vClassId * 0.5);
        finalColor = mix(
            vec3(1.0, 0.6, 0.0),  // Orange
            vec3(1.0, 1.0, 0.2),  // Yellow
            pulse * highlightIntensity
        );
        alpha = min(alpha + 0.2, 1.0);
    } else {
        // Normal points - apply background brightness
        finalColor = vColor * backgroundBrightness;
    }
    
    gl_FragColor = vec4(finalColor, alpha);
}
`;

/**
 * FusionRenderer - Main renderer class for STAC-BUILD
 */
export class FusionRenderer {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            backgroundColor: 0x1a1a2e,
            pointScale: 10,
            minPointSize: 1,
            maxPointSize: 15,
            highlightIntensity: 1.0,
            maxPoints: 10000000, // 5 Million points buffer
            ...options
        };

        // Three.js core
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        // Point cloud
        this.pointCloud = null;
        this.geometry = null;
        this.material = null;

        // State
        this.isRunning = false;
        this.startTime = Date.now();
        this.frameCount = 0;
        this.lastFpsTime = Date.now();
        this.fps = 0;

        // FIX: Cursor to track where to add new points
        this.pointCursor = 0;

        // Chunk offsets for segmentation mapping
        this.chunkOffsets = {};

        // Stats callback
        this.onStats = null;

        this._init();
    }

    _init() {
        const { container, options } = this;
        const width = container.clientWidth;
        const height = container.clientHeight;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(options.backgroundColor);

        // Camera
        this.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
        this.camera.position.set(0, 5, 10);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 0, 0);

        // Grid helper
        const grid = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
        this.scene.add(grid);

        // Axes helper
        const axes = new THREE.AxesHelper(2);
        this.scene.add(axes);

        // Initialize point cloud
        this._initPointCloud();

        // Handle resize
        window.addEventListener('resize', () => this._onResize());
    }

    _initPointCloud() {
        const { options } = this;

        // Pre-allocate geometry buffers
        this.geometry = new THREE.BufferGeometry();

        // Position buffer (x, y, z per vertex)
        const positions = new Float32Array(options.maxPoints * 3);
        this.geometry.setAttribute('position',
            new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage)
        );

        // Color buffer (r, g, b per vertex)
        const colors = new Float32Array(options.maxPoints * 3);
        this.geometry.setAttribute('color',
            new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage)
        );

        // Class ID buffer (1 float per vertex)
        const classIds = new Float32Array(options.maxPoints);
        this.geometry.setAttribute('classId',
            new THREE.BufferAttribute(classIds, 1).setUsage(THREE.DynamicDrawUsage)
        );

        // Initially show 0 points
        this.geometry.setDrawRange(0, 0);
        this.pointCursor = 0; // Reset cursor

        // Shader material
        this.material = new THREE.ShaderMaterial({
            vertexShader,
            fragmentShader,
            uniforms: {
                time: { value: 0 },
                pointScale: { value: options.pointScale },
                minPointSize: { value: options.minPointSize },
                maxPointSize: { value: options.maxPointSize },
                highlightIntensity: { value: options.highlightIntensity },
                backgroundBrightness: { value: 1.0 }
            },
            transparent: true,
            depthWrite: true,
            blending: THREE.NormalBlending
        });

        // Create points object
        this.pointCloud = new THREE.Points(this.geometry, this.material);
        this.pointCloud.frustumCulled = false; // FIX: Prevent disappearing when bounding box is stale
        this.scene.add(this.pointCloud);
    }

    /**
     * Update point cloud from binary data
     * Format: [x, y, z, r, g, b, classId] x N (Float32)
     * FIX: Now ACCUMULATES data instead of overwriting
     */
    updateFromBinary(buffer) {
        if (!buffer || buffer.byteLength === 0) return;

        const floatArray = new Float32Array(buffer);
        const numPointsToAdd = Math.floor(floatArray.length / 7);

        if (numPointsToAdd === 0) return;

        // Check if we have space
        if (this.pointCursor + numPointsToAdd > this.options.maxPoints) {
            console.warn("Max points reached. Resetting buffer (Ring buffer TODO).");
            // Optional: this.clear(); 
            return;
        }

        // Get buffer attributes
        const positions = this.geometry.attributes.position.array;
        const colors = this.geometry.attributes.color.array;
        const classIds = this.geometry.attributes.classId.array;

        // Parse binary data
        let srcIdx = 0;

        // FIX: Start writing at this.pointCursor, not 0
        for (let i = 0; i < numPointsToAdd; i++) {
            const dstIdx = this.pointCursor + i;
            const dstIdx3 = dstIdx * 3;

            // Position (x, y, z)
            positions[dstIdx3] = floatArray[srcIdx++];
            positions[dstIdx3 + 1] = floatArray[srcIdx++];
            positions[dstIdx3 + 2] = floatArray[srcIdx++];

            // Color (r, g, b)
            colors[dstIdx3] = floatArray[srcIdx++];
            colors[dstIdx3 + 1] = floatArray[srcIdx++];
            colors[dstIdx3 + 2] = floatArray[srcIdx++];

            // Class ID
            classIds[dstIdx] = floatArray[srcIdx++];
        }

        // FIX: Advance cursor
        this.pointCursor += numPointsToAdd;

        // Mark buffers as needing update
        this.geometry.attributes.position.needsUpdate = true;
        this.geometry.attributes.color.needsUpdate = true;
        this.geometry.attributes.classId.needsUpdate = true;

        // Update draw range to show ALL points
        this.geometry.setDrawRange(0, this.pointCursor);

        // Only compute bounding sphere occasionally if needed, otherwise it's slow
        // this.geometry.computeBoundingSphere(); 
    }

    /**
     * Register the start of a new chunk to track offsets
     */
    registerChunk(chunkId) {
        this.chunkOffsets[chunkId] = this.pointCursor;
        console.log(`[Renderer] Registered Chunk ${chunkId} at offset ${this.pointCursor}`);
    }

    /**
 * Apply segmentation from JSON with new format
 * data: { chunk_id: int, objects: [{id, label, point_indices}] }
 */
    applySegmentation(data) {
        console.log('[Renderer] applySegmentation called with:', JSON.stringify(data).substring(0, 500));
        const chunkId = data.chunk_id;
        const offset = this.chunkOffsets[chunkId];

        if (offset === undefined) {
            console.warn(`[Renderer] Received segments for unknown chunk ${chunkId}`);
            return;
        }

        const classIds = this.geometry.attributes.classId.array;
        const objects = data.objects || [];
        let modified = false;

        // Store original colors for points that get segmented
        if (!this.originalColors) {
            this.originalColors = new Float32Array(this.geometry.attributes.color.array);
        }

        for (const obj of objects) {
            const objId = obj.id || 1;
            const label = obj.label || "Unknown";
            const indices = obj.point_indices || [];

            // Register this object globally
            const globalObjKey = `${label}_${objId}`;
            if (!this.segmentedObjects) this.segmentedObjects = {};
            if (!this.segmentedObjects[globalObjKey]) {
                this.segmentedObjects[globalObjKey] = {
                    id: objId,
                    label: label,
                    chunks: [],
                    visible: true,  // Default highlighted
                    totalPoints: 0,
                    globalIndices: []  // Store all global indices for toggle
                };
            }
            this.segmentedObjects[globalObjKey].chunks.push(chunkId);
            this.segmentedObjects[globalObjKey].totalPoints += indices.length;

            // Apply classId to points and STORE global indices
            for (const idx of indices) {
                const globalIdx = offset + idx;
                if (globalIdx < this.options.maxPoints) {
                    classIds[globalIdx] = objId;
                    this.segmentedObjects[globalObjKey].globalIndices.push(globalIdx);
                    modified = true;
                }
            }
        }

        if (modified) {
            this.geometry.attributes.classId.needsUpdate = true;
            console.log(`[Renderer] Applied segments for Chunk ${chunkId}:`, objects.map(o => o.label));

            // Notify UI that objects changed
            if (this.onSegmentUpdate) {
                this.onSegmentUpdate(this.segmentedObjects);
            }
        }
    }

    /**
     * Toggle highlight visibility for a specific object
     */
    setObjectHighlight(label, visible) {
        if (!this.segmentedObjects) return;

        const obj = Object.values(this.segmentedObjects).find(o => o.label === label);
        if (!obj || !obj.globalIndices) return;

        obj.visible = visible;

        // Update classId for all stored indices
        const classIds = this.geometry.attributes.classId.array;
        const newClassId = visible ? obj.id : 0;

        for (const globalIdx of obj.globalIndices) {
            if (globalIdx < classIds.length) {
                classIds[globalIdx] = newClassId;
            }
        }

        this.geometry.attributes.classId.needsUpdate = true;
        console.log(`[Renderer] ${visible ? 'Enabled' : 'Disabled'} highlight for "${label}" (${obj.globalIndices.length} points)`);
    }

    /**
     * Set background brightness (0-1, where 1 = full brightness)
     */
    setBackgroundBrightness(brightness) {
        if (this.material && this.material.uniforms.backgroundBrightness) {
            this.material.uniforms.backgroundBrightness.value = brightness;
        }
        console.log(`[Renderer] Background brightness set to ${brightness}`);
    }

    /**
     * Clear the point cloud and segmentation
     */
    clear() {
        this.pointCursor = 0;
        this.geometry.setDrawRange(0, 0);

        // Clear chunk tracking
        this.chunkOffsets = {};

        // Clear segmentation data
        this.segmentedObjects = {};

        // Reset colors to original (if any modifications were made)
        // No need to do anything as geometry is cleared

        // Notify UI to clear segment panel
        if (this.onSegmentUpdate) {
            this.onSegmentUpdate({});
        }

        console.log("Renderer cleared (points + segments)");
    }

    /**
     * Start the render loop
     */
    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        this._animate();
    }

    /**
     * Stop the render loop
     */
    stop() {
        this.isRunning = false;
    }

    _animate() {
        if (!this.isRunning) return;

        requestAnimationFrame(() => this._animate());

        // Update time uniform for shader animations
        const elapsed = (Date.now() - this.startTime) / 1000;
        if (this.material) {
            this.material.uniforms.time.value = elapsed;
        }

        // Update controls
        this.controls.update();

        // Render
        this.renderer.render(this.scene, this.camera);

        // FPS calculation
        this.frameCount++;
        const now = Date.now();
        if (now - this.lastFpsTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastFpsTime = now;

            if (this.onStats) {
                this.onStats({
                    fps: this.fps,
                    points: this.pointCursor // Show accumulated points
                });
            }
        }
    }

    _onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    /**
     * Set camera position
     */
    setCameraPosition(x, y, z) {
        this.camera.position.set(x, y, z);
    }

    /**
     * Look at a point
     */
    lookAt(x, y, z) {
        this.controls.target.set(x, y, z);
        this.camera.lookAt(x, y, z);
    }

    /**
     * Get camera pose matrix
     */
    getCameraPose() {
        return this.camera.matrixWorld.clone();
    }

    /**
     * Dispose resources
     */
    dispose() {
        this.stop();
        this.geometry.dispose();
        this.material.dispose();
        this.renderer.dispose();
        this.container.removeChild(this.renderer.domElement);
    }
}

/**
 * WebSocket connection manager
 */
export class STACConnection {
    constructor(url, renderer) {
        this.url = url;
        this.renderer = renderer;
        this.ws = null;
        this.isConnected = false;
        this.reconnectInterval = 3000;
        this.latency = 0;

        this.onStatusChange = null;
        this.onMessage = null;
    }

    connect() {
        if (this.ws) {
            this.ws.close();
        }

        console.log(`[STAC] Connecting to ${this.url}...`);
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
            console.log('[STAC] Connected');
            this.isConnected = true;
            if (this.onStatusChange) this.onStatusChange(true);
        };

        this.ws.onclose = () => {
            console.log('[STAC] Disconnected');
            this.isConnected = false;
            if (this.onStatusChange) this.onStatusChange(false);

            // Auto reconnect
            setTimeout(() => this.connect(), this.reconnectInterval);
        };

        this.ws.onerror = (error) => {
            console.error('[STAC] WebSocket error:', error);
        };

        this.ws.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                // Binary point cloud data
                const start = performance.now();
                this.renderer.updateFromBinary(event.data);
                this.latency = performance.now() - start;
            } else {
                // JSON message
                try {
                    const data = JSON.parse(event.data);

                    // Handle clear command specifically
                    if (data.type === 'clear' || data.type === 'cleared' || data.type === 'reset') {
                        this.renderer.clear();
                    }

                    if (data.type === 'chunk_start') {
                        this.renderer.registerChunk(data.chunk_id);
                    }

                    if (data.type === 'segmentation') {
                        this.renderer.applySegmentation(data);
                    }

                    if (this.onMessage) this.onMessage(data);
                } catch (e) {
                    console.error('[STAC] JSON parse error:', e);
                }
            }
        };
    }

    send(command) {
        if (this.ws && this.isConnected) {
            this.ws.send(JSON.stringify(command));
        }
    }

    sendPrompt(text) {
        this.send({ type: 'set_prompt', prompt: text });
    }

    sendClear() {
        this.send({ type: 'clear' });
        this.renderer.clear();
    }

    requestStatus() {
        this.send({ type: 'status' });
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

export default FusionRenderer;