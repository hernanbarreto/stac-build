/**
 * SceneShadows — meshes cast and receive shadows; the POINT CLOUD receives them too (USER 2026-10-01,
 * step 1: "mallas que proyectan y reciben + puntos que reciben").
 *
 *  - one sun-like DirectionalLight from above (slightly tilted, as indoor light falls), whose frustum is
 *    fitted every frame to the casters and the ground under them;
 *  - meshes (spheres, the reference human, ShapeR objects, Mesh results, placed objects): three's own
 *    shadow map — castShadow + receiveShadow;
 *  - points: the Potree shader cannot use three's shadow chunks, so the casters are also drawn into a
 *    DEPTH texture from the same light (only objects on CASTER_LAYER: the meshes, never the points),
 *    and the point shader compares each point's depth from the light against it (3x3 PCF).
 * The point cloud does not cast (step 2, optional: it costs a second pass over millions of points).
 */
import * as THREE from 'three'

const CASTER_LAYER = 7
const LIGHT_DIR = new THREE.Vector3(0.35, 1.0, 0.25).normalize()   // from where the light comes
const GROUND_DROP = 3.0          // m below the casters the shadow frustum still reaches (the floor)
const MARGIN = 1.0               // m around the casters
const MAP = 2048                 // shadow map resolution (px)
const BIAS_M = 0.02              // a surface this close to the caster's depth is lit (acne guard), metres

export interface PointShadowUniforms {
    uShadowOn: { value: boolean }
    uShadowDepth: { value: THREE.Texture | null }
    uShadowVP: { value: THREE.Matrix4 }
    uShadowBias: { value: number }
    uShadowTexel: { value: number }
}

export function pointShadowUniforms(): PointShadowUniforms {
    return {
        uShadowOn: { value: false }, uShadowDepth: { value: null }, uShadowVP: { value: new THREE.Matrix4() },
        uShadowBias: { value: 0 }, uShadowTexel: { value: 1 / MAP },
    }
}

export class SceneShadows {
    readonly light: THREE.DirectionalLight
    private rt: THREE.WebGLRenderTarget
    private depthMat = new THREE.MeshDepthMaterial()
    private enabled = true
    private casters: THREE.Object3D[] = []
    private roots: () => THREE.Object3D[] = () => []
    private sinceScan = Infinity

    /** `replaces`: the scene's key light — the sun takes its colour and intensity and stands in for it
     *  while shadows are on, so the meshes are not lit twice */
    constructor(private scene: THREE.Scene, private renderer: THREE.WebGLRenderer, private uniforms: () => PointShadowUniforms[],
                private replaces?: THREE.DirectionalLight) {
        renderer.shadowMap.enabled = true
        renderer.shadowMap.type = THREE.PCFSoftShadowMap
        this.light = new THREE.DirectionalLight(replaces ? replaces.color.getHex() : 0xffffff, replaces ? replaces.intensity : 0.8)
        if (replaces) replaces.visible = false
        this.light.name = 'shadow-sun'
        this.light.castShadow = true
        this.light.shadow.mapSize.set(MAP, MAP)
        this.light.shadow.bias = -0.0005
        this.light.shadow.normalBias = 0.02
        scene.add(this.light, this.light.target)
        this.rt = new THREE.WebGLRenderTarget(MAP, MAP)
        this.rt.depthTexture = new THREE.DepthTexture(MAP, MAP)
        this.rt.depthTexture.type = THREE.UnsignedIntType
        this.light.shadow.camera.layers.enableAll()
    }

    /** the groups whose meshes cast and receive (re-read twice a second) */
    setRoots(fn: () => THREE.Object3D[]) { this.roots = fn }

    setEnabled(on: boolean) {
        this.enabled = on
        this.light.castShadow = on
        this.light.visible = on
        if (this.replaces) this.replaces.visible = !on
        if (!on) for (const u of this.uniforms()) u.uShadowOn.value = false
    }

    get isEnabled() { return this.enabled }

    private scan() {
        const list: THREE.Object3D[] = []
        for (const root of this.roots()) {
            if (!root) continue
            root.traverse(o => {
                const m = o as THREE.Mesh
                if (!m.isMesh) return
                m.castShadow = true
                m.receiveShadow = true
                m.layers.enable(CASTER_LAYER)
                list.push(m)
            })
        }
        this.casters = list
    }

    /** Fit the light to the visible casters, draw their depth for the points. Called every frame. */
    update(dt: number) {
        if (!this.enabled) return
        this.sinceScan += dt
        if (this.sinceScan > 0.5) { this.scan(); this.sinceScan = 0 }
        const box = new THREE.Box3()
        for (const c of this.casters) {
            let vis = true
            for (let p: THREE.Object3D | null = c; p; p = p.parent) if (!p.visible) { vis = false; break }
            if (vis) box.expandByObject(c)
        }
        const us = this.uniforms()
        if (box.isEmpty()) { for (const u of us) u.uShadowOn.value = false; return }
        box.min.y -= GROUND_DROP
        box.expandByScalar(MARGIN)
        const centre = box.getCenter(new THREE.Vector3())
        const r = box.getSize(new THREE.Vector3()).length() / 2
        const cam = this.light.shadow.camera as THREE.OrthographicCamera
        this.light.position.copy(centre).addScaledVector(LIGHT_DIR, r + 10)
        this.light.target.position.copy(centre)
        this.light.target.updateMatrixWorld()
        cam.left = -r; cam.right = r; cam.top = r; cam.bottom = -r
        cam.near = 0.1; cam.far = 2 * r + 20
        cam.updateProjectionMatrix()
        this.light.updateMatrixWorld()
        cam.position.copy(this.light.position)
        cam.lookAt(centre)
        cam.updateMatrixWorld()
        // the casters' depth from the light, for the point shader (meshes only: CASTER_LAYER)
        const prevTarget = this.renderer.getRenderTarget()
        const prevOverride = this.scene.overrideMaterial
        const prevBg = this.scene.background
        const prevMask = cam.layers.mask
        cam.layers.set(CASTER_LAYER)
        this.scene.overrideMaterial = this.depthMat
        this.scene.background = null
        this.renderer.setRenderTarget(this.rt)
        this.renderer.clear()
        this.renderer.render(this.scene, cam)
        this.renderer.setRenderTarget(prevTarget)
        this.scene.overrideMaterial = prevOverride
        this.scene.background = prevBg
        cam.layers.mask = prevMask
        const vp = new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse)
        for (const u of us) {
            u.uShadowOn.value = true
            u.uShadowDepth.value = this.rt.depthTexture
            u.uShadowVP.value.copy(vp)
            u.uShadowBias.value = BIAS_M / (cam.far - cam.near)
        }
    }

    dispose() {
        this.scene.remove(this.light, this.light.target)
        if (this.replaces) this.replaces.visible = true
        this.rt.dispose(); this.depthMat.dispose()
    }
}

/** GLSL for the point shader: declarations (both stages) + the fragment's shade factor. */
export const POINT_SHADOW_VERT_DECL = `
  uniform mat4 uShadowVP;
  varying vec4 vShadowPos;
`
export const POINT_SHADOW_VERT_MAIN = `
    vShadowPos = uShadowVP * vec4(vWorldPos, 1.0);
`
export const POINT_SHADOW_FRAG_DECL = `
  uniform bool uShadowOn;
  uniform sampler2D uShadowDepth;
  uniform float uShadowBias;
  uniform float uShadowTexel;
  varying vec4 vShadowPos;
  float pointShadow() {
    if (!uShadowOn) return 1.0;
    vec3 sc = vShadowPos.xyz / vShadowPos.w * 0.5 + 0.5;
    if (sc.x <= 0.0 || sc.x >= 1.0 || sc.y <= 0.0 || sc.y >= 1.0 || sc.z >= 1.0) return 1.0;
    float lit = 0.0;
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
      float d = texture2D(uShadowDepth, sc.xy + vec2(float(i), float(j)) * uShadowTexel).r;
      lit += (sc.z - uShadowBias <= d) ? 1.0 : 0.0;
    }
    return mix(0.45, 1.0, lit / 9.0);
  }
`
