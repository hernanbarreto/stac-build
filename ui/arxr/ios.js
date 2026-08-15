// STAC-Builder — iPhone XR viewer on the 8th Wall engine (static/ar/ios.html).
//
// iOS Safari has NO WebXR (2026: the flag exists but is non-functional), so
// this page uses the 8th Wall engine binary (self-hosted, free since the
// 02/2026 Niantic open release): its own SLAM world tracking over the live
// camera feed with ABSOLUTE (metric) scale, no App Clips, no accounts.
//
// Flow: ?session=<id> → engine starts (camera + tracking) → reticle follows
// the screen-center hit test on detected surfaces → tap places the session
// mesh (1:10 tabletop first, scale button cycles to metric 1:1) → measurement
// tools (distance / angle / volume, computed in METRIC model space).
//
// Built by ui/arxr/build.sh → static/ar/ios-app.js. The engine is provisioned
// from npm @8thwall/engine-binary into static/ar/xr8/ (git-ignored).
//
// Hernán Barreto - Ingerop IN3 Session IV - STAC

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh';

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

const $ = (id) => document.getElementById(id);
const API = '';
const sessionId = new URLSearchParams(location.search).get('session');

function tele(event, data = {}) {
  try {
    fetch(`${API}/api/ar/log`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event: `ios-${event}`, session: sessionId, ...data }),
    }).catch(() => {});
  } catch { /* telemetry must never break the app */ }
}
window.addEventListener('error', (e) =>
  tele('js-error', { msg: String(e.message), src: `${e.filename}:${e.lineno}` }));
window.addEventListener('unhandledrejection', (e) =>
  tele('promise-rejection', { msg: String(e.reason).slice(0, 300) }));

let toastTimer = null;
function toast(msg, ms = 2600) {
  const t = $('toast');
  t.textContent = msg;
  t.style.display = 'block';
  clearTimeout(toastTimer);
  if (ms > 0) toastTimer = setTimeout(() => { t.style.display = 'none'; }, ms);
}

// ── state ─────────────────────────────────────────────────────────────────────

let scene, camera;                 // provided by XR8.Threejs
let contentGroup, modelGroup, measureGroup, reticle;
let placed = false;
let currentTool = 'move';
let pending = [];
const SCALES = [[0.1, '1:10'], [1, '1:1'], [0.02, '1:50']];
let scaleIdx = 0;
let lastAnchor = null;
const raycaster = new THREE.Raycaster();
raycaster.firstHitOnly = true;
let raycastTargets = [];

// ── mesh loading ──────────────────────────────────────────────────────────────

async function loadMesh() {
  const loader = new GLTFLoader();
  loader.setMeshoptDecoder(MeshoptDecoder);
  const gltf = await loader.loadAsync(
    `${API}/api/ar/mesh/${encodeURIComponent(sessionId)}`);
  const root = gltf.scene;
  root.traverse((o) => {
    if (o.isMesh) {
      o.geometry.computeBoundsTree();
      raycastTargets.push(o);
      // fullbright: photogrammetry textures carry the lighting already
      o.material = new THREE.MeshBasicMaterial({
        map: o.material?.map || null,
        color: o.material?.color?.clone() || new THREE.Color(0xffffff),
      });
    }
  });
  contentGroup.add(root);
  // upright transform for sessions whose in-pipeline orientation was refused
  try {
    const r = await fetch(`${API}/api/ar/sessions`);
    const d = await r.json();
    const s = d.sessions?.find((x) => x.id === sessionId);
    if (s?.floor_transform) {
      contentGroup.matrixAutoUpdate = false;
      contentGroup.matrix.set(...s.floor_transform);
      contentGroup.matrixWorldNeedsUpdate = true;
    }
  } catch { /* orientation stays as baked */ }
}

// ── placement (metric model space; floor plane y=0 is pipeline-calibrated) ────

function placeContentAt(x, y, z) {
  contentGroup.visible = true;
  placed = true;
  const box = new THREE.Box3().setFromObject(contentGroup);
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3());
  modelGroup.position.x += x - c.x;
  modelGroup.position.z += z - c.z;
  modelGroup.position.y = y;       // model floor (y=0, calibrated) onto the hit
  lastAnchor = new THREE.Vector3(x, y, z);
}

function setScaleIdx(i) {
  scaleIdx = i % SCALES.length;
  modelGroup.scale.setScalar(SCALES[scaleIdx][0]);
  $('btn-scale').textContent = SCALES[scaleIdx][1];
  if (lastAnchor) placeContentAt(lastAnchor.x, lastAnchor.y, lastAnchor.z);
}

// ── measurement (identical math to the main viewer, metric model space) ──────

function fmt(m) { return m >= 1 ? `${m.toFixed(2)} m` : `${(m * 100).toFixed(1)} cm`; }

function marker(p, color = 0xffc107) {
  const s = new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 12),
                           new THREE.MeshBasicMaterial({ color }));
  s.position.copy(p);
  measureGroup.add(s);
}

function line(points, color = 0xffc107) {
  measureGroup.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color })));
}

function label(text, p) {
  const pad = 8, fs = 34;
  const cv = document.createElement('canvas');
  const cx = cv.getContext('2d');
  cx.font = `600 ${fs}px system-ui`;
  cv.width = cx.measureText(text).width + pad * 2;
  cv.height = fs + pad * 2;
  cx.font = `600 ${fs}px system-ui`;
  cx.fillStyle = 'rgba(12,16,20,0.85)';
  cx.fillRect(0, 0, cv.width, cv.height);
  cx.fillStyle = '#ffd866';
  cx.textBaseline = 'middle';
  cx.fillText(text, pad, cv.height / 2);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(cv), depthTest: false }));
  sp.renderOrder = 999;
  const h = 0.12;
  sp.scale.set(h * cv.width / cv.height, h, 1);
  sp.position.copy(p);
  measureGroup.add(sp);
}

function addMeasurePoint(p) {
  if (currentTool === 'dist') {
    pending.push(p); marker(p);
    if (pending.length === 2) {
      const [a, b] = pending;
      line([a, b]);
      label(fmt(a.distanceTo(b)), a.clone().add(b).multiplyScalar(0.5));
      pending = [];
    }
  } else if (currentTool === 'angle') {
    pending.push(p); marker(p, 0x79c0ff);
    if (pending.length === 3) {
      const [a, v, b] = pending;
      line([a, v, b], 0x79c0ff);
      const u1 = a.clone().sub(v).normalize(), u2 = b.clone().sub(v).normalize();
      const deg = THREE.MathUtils.radToDeg(
        Math.acos(THREE.MathUtils.clamp(u1.dot(u2), -1, 1)));
      label(`${deg.toFixed(1)}°`,
            v.clone().addScaledVector(u1.clone().add(u2).normalize(), 0.25));
      pending = [];
    }
  } else if (currentTool === 'vol') {
    pending.push(p); marker(p, 0x4ade80);
    if (pending.length === 3) {
      const [a, b, c] = pending;
      const y0 = Math.min(a.y, b.y);
      const h = Math.max(Math.abs(c.y - y0), 0.05);
      const min = new THREE.Vector3(Math.min(a.x, b.x), y0, Math.min(a.z, b.z));
      const size = new THREE.Vector3(Math.max(a.x, b.x), y0 + h,
                                     Math.max(a.z, b.z)).sub(min);
      const vol = size.x * size.y * size.z;
      const geo = new THREE.BoxGeometry(size.x, size.y, size.z);
      const box = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
        color: 0x4ade80, transparent: true, opacity: 0.18, depthWrite: false }));
      box.position.copy(min).add(size.clone().multiplyScalar(0.5));
      measureGroup.add(box);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0x4ade80 }));
      edges.position.copy(box.position);
      measureGroup.add(edges);
      label(`${size.x.toFixed(2)}×${size.z.toFixed(2)}×${size.y.toFixed(2)} m = `
        + `${vol.toFixed(2)} m³`, box.position.clone().setY(min.y + size.y + 0.1));
      pending = [];
    }
  }
}

// ── 8th Wall pipeline module ─────────────────────────────────────────────────

function stacPipelineModule() {
  return {
    name: 'stac-xr',
    onStart: () => {
      const s = window.XR8.Threejs.xrScene();
      scene = s.scene;
      camera = s.camera;
      camera.position.set(0, 1.6, 0);          // standing eye height (meters)

      scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.0));
      modelGroup = new THREE.Group();
      contentGroup = new THREE.Group();
      measureGroup = new THREE.Group();
      modelGroup.add(contentGroup, measureGroup);
      contentGroup.visible = false;            // NOTHING until the user places
      scene.add(modelGroup);

      reticle = new THREE.Mesh(
        new THREE.RingGeometry(0.07, 0.09, 32).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: 0x4ade80, depthTest: false }));
      reticle.renderOrder = 998;
      reticle.visible = false;
      scene.add(reticle);

      setScaleIdx(0);
      loadMesh().then(() => {
        $('splash').style.display = 'none';
        toast('Aim the circle at the floor and tap to place');
        tele('ready', {});
      }).catch((e) => {
        $('splash-msg').textContent = `Mesh load failed: ${e.message || e}`;
        tele('mesh-error', { msg: String(e.message || e) });
      });
    },
    onUpdate: () => {
      if (!scene || currentTool !== 'move') { if (reticle) reticle.visible = false; return; }
      const hits = window.XR8.XrController.hitTest(0.5, 0.5,
        ['DETECTED_SURFACE', 'ESTIMATED_SURFACE', 'FEATURE_POINT']);
      const h = hits && hits[0];
      if (h) {
        reticle.position.set(h.position.x, h.position.y, h.position.z);
        const d = Math.max(reticle.position.distanceTo(camera.position) / 6, 0.6);
        reticle.scale.setScalar(d);
        reticle.visible = true;
      } else {
        reticle.visible = false;
      }
    },
    listeners: [{
      event: 'reality.error',
      process: (e) => {
        tele('reality-error', { msg: String(e?.detail || e) });
        $('splash').style.display = 'flex';
        $('splash-msg').textContent =
          'Camera/tracking failed to start — allow camera + motion access and reload';
      },
    }],
  };
}

// ── taps ─────────────────────────────────────────────────────────────────────

function bindTaps() {
  const cv = $('camerafeed');
  let down = null;
  cv.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) down = [e.touches[0].clientX, e.touches[0].clientY];
  });
  cv.addEventListener('touchend', (e) => {
    if (!down) return;
    const t = e.changedTouches[0];
    const moved = Math.hypot(t.clientX - down[0], t.clientY - down[1]);
    down = null;
    if (moved > 10 || !scene) return;
    if (currentTool === 'move') {
      if (reticle.visible) {
        const p = reticle.position;
        placeContentAt(p.x, p.y, p.z);
        toast('Placed — scale button for 1:1, tools below to measure');
        tele('placed', { scale: SCALES[scaleIdx][1] });
      }
      return;
    }
    if (!placed) { toast('Place the model first (Move + tap)'); return; }
    const ndc = new THREE.Vector2((t.clientX / innerWidth) * 2 - 1,
                                  -(t.clientY / innerHeight) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObjects(raycastTargets, false);
    if (!hits.length) { toast('No mesh under the tap'); return; }
    addMeasurePoint(modelGroup.worldToLocal(hits[0].point.clone())
      .multiplyScalar(1));  // model space is metric; scale handled by group
  });
}

// ── UI ───────────────────────────────────────────────────────────────────────

function wireUI() {
  $('v-title').textContent = sessionId || 'STAC XR';
  $('btn-back').onclick = () => { location.href = 'index.html'; };
  $('btn-scale').onclick = () => setScaleIdx(scaleIdx + 1);
  $('btn-recenter').onclick = () => {
    window.XR8?.XrController.recenter();
    toast('Tracking recentered');
  };
  $('btn-clear').onclick = () => {
    pending = [];
    while (measureGroup?.children.length) measureGroup.children.pop();
  };
  document.querySelectorAll('#toolbar [data-tool]').forEach((b) => {
    b.onclick = () => {
      currentTool = b.dataset.tool;
      pending = [];
      document.querySelectorAll('#toolbar [data-tool]').forEach((x) =>
        x.classList.toggle('active', x === b));
      toast({ move: 'Aim the circle and tap to place',
              dist: 'Tap two points on the mesh',
              angle: 'Tap 3 points (vertex second)',
              vol: 'Tap 2 base corners, then a height point' }[currentTool]);
    };
  });
}

// ── boot ─────────────────────────────────────────────────────────────────────

function startEngine() {
  const XR8 = window.XR8;
  tele('engine-loaded', { version: XR8.version || '?' });
  XR8.XrController.configure({ scale: 'absolute' });   // METRIC world units
  XR8.addCameraPipelineModules([
    XR8.GlTextureRenderer.pipelineModule(),            // camera feed
    XR8.Threejs.pipelineModule(),                      // three scene driven by SLAM
    XR8.XrController.pipelineModule(),                 // 6DoF world tracking
    stacPipelineModule(),
  ]);
  XR8.run({ canvas: $('camerafeed') });
}

if (!sessionId) {
  $('splash-msg').textContent = 'No session selected — go back to the list';
} else {
  wireUI();
  bindTaps();
  tele('boot', { ua: navigator.userAgent });
  if (window.XR8) startEngine();
  else window.addEventListener('xrloaded', startEngine);
}
