/**
 * ViewCube (USER 2026-10-04: "el famoso cubo profesional" instead of the T/F/R/L/B/I buttons).
 *
 * A small three.js scene of its own: a labelled cube that follows the main camera's orientation,
 * whose 26 pieces (6 faces, 12 edges, 8 corners) snap the main view to that direction on click,
 * and which orbits the main camera when dragged; a compass ring (N/E/S/W, rotating with the view's
 * yaw) and a home button under it. It only draws and reports — the main camera lives in Viewport.
 */
import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { Home } from 'lucide-react'
import { Tooltip } from '../ui/Tooltip'
import { useT } from '../../i18n'

export interface ViewCubeProps {
  /** the main camera's world quaternion (null while there is no scene) */
  getQuaternion: () => THREE.Quaternion | null
  /** snap the main view to look along -dir (dir = where the camera sits, as a unit vector in world axes) */
  onOrient: (dir: [number, number, number]) => void
  /** orbit the main camera by azimuth / elevation deltas in radians */
  onOrbit: (dAz: number, dEl: number) => void
  onHome: () => void
  size?: number
}

const FACES: Array<{ key: 'top' | 'bottom' | 'front' | 'back' | 'right' | 'left'; n: [number, number, number] }> = [
  { key: 'top', n: [0, 1, 0] }, { key: 'bottom', n: [0, -1, 0] }, { key: 'front', n: [0, 0, 1] },
  { key: 'back', n: [0, 0, -1] }, { key: 'right', n: [1, 0, 0] }, { key: 'left', n: [-1, 0, 0] },
]

function faceTexture(label: string, fg: string, bg: string): THREE.CanvasTexture {
  const c = document.createElement('canvas'); c.width = 256; c.height = 256
  const g = c.getContext('2d')!
  g.fillStyle = bg; g.fillRect(0, 0, 256, 256)
  g.strokeStyle = 'rgba(0,0,0,0.25)'; g.lineWidth = 6; g.strokeRect(3, 3, 250, 250)
  g.fillStyle = fg; g.textAlign = 'center'; g.textBaseline = 'middle'
  g.font = `600 ${label.length > 5 ? 54 : 64}px ui-sans-serif, system-ui, sans-serif`
  g.fillText(label.toUpperCase(), 128, 132)
  const t = new THREE.CanvasTexture(c); t.anisotropy = 4; t.colorSpace = THREE.SRGBColorSpace
  return t
}

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}

export function ViewCube({ getQuaternion, onOrient, onOrbit, onHome, size = 112 }: ViewCubeProps) {
  const t = useT()
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const compassRef = useRef<HTMLDivElement | null>(null)
  const cbRef = useRef({ getQuaternion, onOrient, onOrbit, onHome })
  cbRef.current = { getQuaternion, onOrient, onOrbit, onHome }
  const labels = useRef<Record<string, string>>({})
  labels.current = { top: t('view.top'), bottom: t('view.bottom'), front: t('view.front'), back: t('view.back'),
                     right: t('view.right'), left: t('view.left') }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setSize(size, size, false)
    const scene = new THREE.Scene()
    const cam = new THREE.OrthographicCamera(-1.05, 1.05, 1.05, -1.05, 0.1, 20)
    scene.add(new THREE.AmbientLight(0xffffff, 0.85))
    const key = new THREE.DirectionalLight(0xffffff, 0.5); key.position.set(2, 3, 4); scene.add(key)

    const fg = cssVar('--text-1', '#e8e8e8')
    const faceBg = cssVar('--surface-3', '#3a3f47')
    const hoverCol = new THREE.Color(cssVar('--brand-orange', '#f28c28'))
    const edgeCol = new THREE.Color(cssVar('--surface-4', '#4a515b'))
    const pieces: THREE.Mesh[] = []
    const H = 0.5, F = 0.36        // half side; half of the flat face area (the rest is edge / corner)
    const E = H - F                // edge thickness
    // faces
    for (const f of FACES) {
      const geo = new THREE.PlaneGeometry(2 * F, 2 * F)
      const mat = new THREE.MeshLambertMaterial({ map: faceTexture(labels.current[f.key], fg, faceBg) })
      const m = new THREE.Mesh(geo, mat)
      const n = new THREE.Vector3(...f.n)
      m.position.copy(n).multiplyScalar(H + 0.001)
      m.lookAt(n.clone().multiplyScalar(2))
      if (f.key === 'top') m.rotation.z = 0                                   // text reads from the front
      if (f.key === 'top') { m.rotation.set(-Math.PI / 2, 0, 0) }
      if (f.key === 'bottom') { m.rotation.set(Math.PI / 2, 0, 0) }
      m.userData = { dir: f.n, kind: 'face' }
      scene.add(m); pieces.push(m)
    }
    // edges and corners: boxes placed on the cube's frame
    const edgeMat = () => new THREE.MeshLambertMaterial({ color: edgeCol })
    const axes = [-1, 0, 1]
    for (const x of axes) for (const y of axes) for (const z of axes) {
      const nz = Math.abs(x) + Math.abs(y) + Math.abs(z)
      if (nz < 2) continue                                                    // 0 = centre, 1 = faces
      const sx = x === 0 ? 2 * F : E, sy = y === 0 ? 2 * F : E, sz = z === 0 ? 2 * F : E
      const m = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), edgeMat())
      m.position.set(x * (H - E / 2), y * (H - E / 2), z * (H - E / 2))
      const d = new THREE.Vector3(x, y, z).normalize()
      m.userData = { dir: [d.x, d.y, d.z], kind: nz === 2 ? 'edge' : 'corner' }
      scene.add(m); pieces.push(m)
    }
    // the faces' own flat body (so the cube is solid behind the labels)
    const body = new THREE.Mesh(new THREE.BoxGeometry(2 * F, 2 * F, 2 * F).scale(1, 1, 1), new THREE.MeshLambertMaterial({ color: edgeCol }))
    scene.add(body)

    const ray = new THREE.Raycaster()
    const ndc = new THREE.Vector2()
    let hovered: THREE.Mesh | null = null
    const setHover = (m: THREE.Mesh | null) => {
      if (hovered === m) return
      if (hovered) { const mat = hovered.material as THREE.MeshLambertMaterial; mat.emissive.setRGB(0, 0, 0) }
      hovered = m
      if (hovered) { const mat = hovered.material as THREE.MeshLambertMaterial; mat.emissive.copy(hoverCol).multiplyScalar(0.55) }
      canvas.style.cursor = hovered ? 'pointer' : 'grab'
      dirty = true
    }
    const pick = (ev: PointerEvent): THREE.Mesh | null => {
      const r = canvas.getBoundingClientRect()
      ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1)
      ray.setFromCamera(ndc, cam)
      const hit = ray.intersectObjects(pieces, false)[0]
      return hit ? (hit.object as THREE.Mesh) : null
    }

    let down: { x: number; y: number; moved: boolean } | null = null
    const onDown = (ev: PointerEvent) => { down = { x: ev.clientX, y: ev.clientY, moved: false }; canvas.setPointerCapture(ev.pointerId) }
    const onMove = (ev: PointerEvent) => {
      if (down) {
        const dx = ev.clientX - down.x, dy = ev.clientY - down.y
        if (Math.abs(dx) + Math.abs(dy) > 2) {
          down.moved = true
          cbRef.current.onOrbit(-dx * 0.012, -dy * 0.012)
          down.x = ev.clientX; down.y = ev.clientY
          canvas.style.cursor = 'grabbing'
        }
        return
      }
      setHover(pick(ev))
    }
    const onUp = (ev: PointerEvent) => {
      if (!down) return
      const wasClick = !down.moved
      down = null
      canvas.style.cursor = hovered ? 'pointer' : 'grab'
      if (wasClick) {
        const m = pick(ev)
        if (m) cbRef.current.onOrient(m.userData.dir as [number, number, number])
      }
    }
    const onLeave = () => { if (!down) setHover(null) }
    canvas.addEventListener('pointerdown', onDown)
    canvas.addEventListener('pointermove', onMove)
    canvas.addEventListener('pointerup', onUp)
    canvas.addEventListener('pointerleave', onLeave)

    let dirty = true
    const last = new THREE.Quaternion(1, 1, 1, 1)
    const fwd = new THREE.Vector3()
    let raf = 0
    const tick = () => {
      raf = requestAnimationFrame(tick)
      const q = cbRef.current.getQuaternion()
      if (q && (!q.equals(last) || dirty)) {
        last.copy(q)
        cam.quaternion.copy(q)
        cam.position.set(0, 0, 3).applyQuaternion(q)
        cam.updateMatrixWorld()
        // the compass turns with the view's yaw (N = world -Z, the back face)
        fwd.set(0, 0, -1).applyQuaternion(q)
        const yaw = Math.atan2(fwd.x, -fwd.z)
        if (compassRef.current) compassRef.current.style.setProperty('--yaw', `${(yaw * 180) / Math.PI}deg`)
        renderer.render(scene, cam)
        dirty = false
      }
    }
    tick()
    return () => {
      cancelAnimationFrame(raf)
      canvas.removeEventListener('pointerdown', onDown)
      canvas.removeEventListener('pointermove', onMove)
      canvas.removeEventListener('pointerup', onUp)
      canvas.removeEventListener('pointerleave', onLeave)
      for (const m of pieces) { m.geometry.dispose(); const mat = m.material as THREE.MeshLambertMaterial; mat.map?.dispose(); mat.dispose() }
      body.geometry.dispose(); (body.material as THREE.Material).dispose()
      renderer.dispose()
    }
  }, [size])

  return (
    <div className="stac-viewcube" role="group" aria-label={t('view.cube')} style={{ '--cube-px': `${size}px` } as React.CSSProperties}>
      <canvas ref={canvasRef} className="stac-viewcube__canvas" width={size} height={size} aria-label={t('view.cube')} />
      <div ref={compassRef} className="stac-viewcube__compass" aria-hidden>
        <span className="stac-viewcube__dir stac-viewcube__dir--n">{t('view.north')}</span>
        <span className="stac-viewcube__dir stac-viewcube__dir--e">{t('view.east')}</span>
        <span className="stac-viewcube__dir stac-viewcube__dir--s">{t('view.south')}</span>
        <span className="stac-viewcube__dir stac-viewcube__dir--w">{t('view.west')}</span>
      </div>
      <Tooltip label={t('toolbar.resetView')} side="left">
        <button type="button" className="stac-viewcube__home" onClick={onHome} aria-label={t('toolbar.resetView')}>
          <Home aria-hidden />
        </button>
      </Tooltip>
    </div>
  )
}
