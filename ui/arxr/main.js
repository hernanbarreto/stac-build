// STAC-Builder — WebXR AR viewer (phone, over the tailnet).
//
// Session picker → metric mesh (scene.glb, meshopt+WebP) and/or decimated
// point cloud (ARC1 binary) → WebXR immersive-ar placement with hit-test when
// the browser offers it → measurement tools (distance / angle / volume box,
// computed in MODEL space so they stay metric at any display scale) → Spatial
// AI chat against POST /api/spatial_qa with best-effort 3D markers from the
// tool traces.
//
// Non-XR fallback: the same scene with OrbitControls — every tool works there
// too, so the app is fully testable from a desktop browser.
//
// Built with esbuild (ui/arxr/build.sh) → static/ar/app.js.
//
// Hernán Barreto - Ingerop IN3 Session IV - STAC

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh';

// BVH-accelerated raycast: the TSDF mesh has millions of triangles — the
// stock raycaster takes seconds per tap on a phone, the BVH takes ~1 ms.
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

const $ = (id) => document.getElementById(id);
const API = '';                       // same origin

// remote telemetry: XR browsers have no devtools — report capabilities and
// errors to the pod (POST /api/ar/log → logs/ar_client.jsonl)
function tele(event, data = {}) {
  try {
    fetch(`${API}/api/ar/log`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event, ...data }),
    }).catch(() => {});
  } catch { /* never break the app for telemetry */ }
}
window.addEventListener('error', (e) =>
  tele('js-error', { msg: String(e.message), src: `${e.filename}:${e.lineno}` }));
window.addEventListener('unhandledrejection', (e) =>
  tele('promise-rejection', { msg: String(e.reason).slice(0, 300) }));
tele('boot', { ua: navigator.userAgent, hasXR: !!navigator.xr });

// ── UI helpers ────────────────────────────────────────────────────────────────

let toastTimer = null;
function toast(msg, ms = 2600) {
  const t = $('toast');
  t.textContent = msg;
  t.style.display = 'block';
  clearTimeout(toastTimer);
  if (ms > 0) toastTimer = setTimeout(() => { t.style.display = 'none'; }, ms);
}
function loading(on, msg = 'Loading…') {
  $('loading').style.display = on ? 'flex' : 'none';
  $('loading-msg').textContent = msg;
}

// ── session picker ────────────────────────────────────────────────────────────

async function loadSessions() {
  const cards = $('cards');
  try {
    const r = await fetch(`${API}/api/ar/sessions`);
    const data = await r.json();
    if (!data.sessions?.length) {
      cards.innerHTML = '<div class="empty">No reconstructions with mesh or cloud yet.</div>';
      return;
    }
    cards.innerHTML = '';
    for (const s of data.sessions) {
      const card = document.createElement('div');
      card.className = 'card';
      const mb = s.mesh_bytes ? ` (${(s.mesh_bytes / 1e6).toFixed(0)} MB)` : '';
      card.innerHTML = `
        <div class="name">${s.id}</div>
        <div class="badges">
          <span class="badge ${s.has_mesh ? 'on' : ''}">mesh${s.has_mesh ? mb : ' ✕'}</span>
          <span class="badge ${s.has_cloud ? 'on' : ''}">cloud${s.has_cloud ? ` · ${s.cloud_source}` : ' ✕'}</span>
          <span class="badge ${s.has_ai ? 'on' : ''}">AI ${s.has_ai ? '✦' : '✕ (no instance store)'}</span>
        </div>
        <div class="row">
          <label><input type="checkbox" class="ck-mesh" ${s.has_mesh ? 'checked' : 'disabled'}> mesh</label>
          <label><input type="checkbox" class="ck-cloud" ${s.has_cloud && !s.has_mesh ? 'checked' : ''} ${s.has_cloud ? '' : 'disabled'}> cloud</label>
          <button class="open primary">Open</button>
        </div>`;
      card.querySelector('.open').onclick = () => {
        const wantMesh = card.querySelector('.ck-mesh').checked;
        const wantCloud = card.querySelector('.ck-cloud').checked;
        if (!wantMesh && !wantCloud) { toast('Pick mesh, cloud, or both'); return; }
        openViewer(s, wantMesh, wantCloud);
      };
      cards.appendChild(card);
    }
  } catch (e) {
    cards.innerHTML = `<div class="empty">Backend unreachable: ${e}</div>`;
  }
}

// ── three.js scene ────────────────────────────────────────────────────────────

let renderer, scene, camera, controls;
let modelGroup;          // placed/scaled in AR; children live in METRIC model space
let contentGroup;        // mesh + cloud
let measureGroup;        // measurement geometry (model space)
let aiGroup;             // AI trace markers (model space)
let reticle, hitTestSource = null, xrRefSpace = null, xrRefType = 'local-floor';
let hudGroup = null, hudButtons = [], hudStatus = null, xrBlendMode = '';
let session = null;      // current STAC session entry
let floorMatrix = null;  // upright transform for sessions whose in-pipeline
                         // orientation was refused (server sends 4x4 row-major)
let raycastTargets = [];
let displayScale = 1;    // 1, 0.1, 0.02
const SCALES = [[1, '1:1'], [0.1, '1:10'], [0.02, '1:50']];
let scaleIdx = 0;
let currentTool = 'move';
let fullbright = true;
const savedMaterials = new Map();

function initThree() {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setClearColor(0x000000, 0);   // AR passthrough needs a transparent clear
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.xr.enabled = true;
  $('canvas-wrap').appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.02, 300);
  camera.position.set(4, 3, 4);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.1));
  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(3, 10, 4);
  scene.add(dir);

  modelGroup = new THREE.Group();
  contentGroup = new THREE.Group();
  measureGroup = new THREE.Group();
  aiGroup = new THREE.Group();
  modelGroup.add(contentGroup, measureGroup, aiGroup);
  scene.add(modelGroup);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 1, 0);

  reticle = new THREE.Mesh(
    new THREE.RingGeometry(0.07, 0.09, 32).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: 0x4ade80, depthTest: false }));
  reticle.renderOrder = 998;          // never hidden behind model geometry
  reticle.matrixAutoUpdate = false;
  reticle.visible = false;
  scene.add(reticle);

  window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  renderer.setAnimationLoop(onFrame);
  bindTapPicking();
  bindPinchFov();
}

const _centerNdc = new THREE.Vector2(0, 0);

function onFrame(_t, frame) {
  // Cam AR reticle: the screen center projected onto the virtual floor — aim
  // with the phone, tap to place (Move tool)
  if (camAR.on) {
    if (currentTool === 'move') {
      raycaster.setFromCamera(_centerNdc, camera);
      const hit = new THREE.Vector3();
      if (raycaster.ray.intersectPlane(_floorPlane, hit)
          && hit.distanceTo(camera.position) < 60) {
        reticle.matrixAutoUpdate = true;
        reticle.position.copy(hit);
        const d = Math.max(hit.distanceTo(camera.position) / 6, 0.6);
        reticle.scale.setScalar(d);          // keep it visible at distance
        reticle.visible = true;
      } else reticle.visible = false;
    } else reticle.visible = false;
  }
  if (frame && hitTestSource) {
    const hits = frame.getHitTestResults(hitTestSource);
    if (hits.length) {
      const pose = hits[0].getPose(xrRefSpace);
      reticle.visible = currentTool === 'move';
      reticle.matrix.fromArray(pose.transform.matrix);
    } else {
      reticle.visible = false;
    }
  }
  updateHudPlacement();
  controls.enabled = !renderer.xr.isPresenting;
  renderer.render(scene, camera);
}

// ── asset loading ─────────────────────────────────────────────────────────────

async function loadMesh(id) {
  const loader = new GLTFLoader();
  loader.setMeshoptDecoder(MeshoptDecoder);
  const gltf = await loader.loadAsync(`${API}/api/ar/mesh/${encodeURIComponent(id)}`);
  const root = gltf.scene;
  root.traverse((o) => {
    if (o.isMesh) {
      o.geometry.computeBoundsTree();
      raycastTargets.push(o);
      savedMaterials.set(o, o.material);
    }
  });
  contentGroup.add(root);
  applyLighting();
}

async function loadCloud(id) {
  const r = await fetch(`${API}/api/ar/cloud/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error(`cloud HTTP ${r.status}`);
  const buf = await r.arrayBuffer();
  const dv = new DataView(buf);
  if (dv.getUint32(0, false) !== 0x41524331) throw new Error('bad ARC1 magic');
  const n = dv.getUint32(4, true);
  const xyz = new Float32Array(buf, 32, n * 3);
  const rgbU8 = new Uint8Array(buf, 32 + n * 12, n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) col[i] = rgbU8[i] / 255;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const pts = new THREE.Points(g, new THREE.PointsMaterial({
    size: 0.014, vertexColors: true, sizeAttenuation: true }));
  pts.userData.isCloud = true;
  contentGroup.add(pts);
  raycastTargets.push(pts);
}

function applyLighting() {
  contentGroup.traverse((o) => {
    if (!o.isMesh) return;
    if (fullbright) {
      if (!o.userData.fbMat) {
        const src = savedMaterials.get(o);
        o.userData.fbMat = new THREE.MeshBasicMaterial({
          map: src && src.map ? src.map : null,
          vertexColors: !!(src && src.vertexColors),
          color: src && src.color ? src.color.clone() : new THREE.Color(0xffffff),
        });
      }
      o.material = o.userData.fbMat;
    } else if (savedMaterials.has(o)) {
      o.material = savedMaterials.get(o);
    }
  });
}

async function openViewer(s, wantMesh, wantCloud) {
  session = s;
  $('home').style.display = 'none';
  $('viewer').style.display = 'block';
  $('v-title').textContent = s.id;
  if (!renderer) initThree();
  clearContent();
  loading(true, `Loading ${s.id}…`);
  try {
    if (wantMesh) { loading(true, 'Loading mesh…'); await loadMesh(s.id); }
    if (wantCloud) { loading(true, 'Loading point cloud…'); await loadCloud(s.id); }
    // upright the content when the pipeline's own orientation gate refused
    // (measurements + AR floor placement need gravity-aligned Y)
    floorMatrix = null;
    if (s.floor_transform) {
      floorMatrix = new THREE.Matrix4().set(...s.floor_transform);
      contentGroup.matrixAutoUpdate = false;
      contentGroup.matrix.copy(floorMatrix);
      contentGroup.matrixWorldNeedsUpdate = true;
    }
    frameContent();
    setTool('move');
    toast(s.has_ai ? 'Tip: the AI ✦ button answers with real measurements'
                   : 'AI disabled: this session has no instance store (run segmentation)');
  } catch (e) {
    toast(`Load failed: ${e.message || e}`, 6000);
  } finally {
    loading(false);
  }
  setupARButton();
}

function clearContent() {
  for (const grp of [contentGroup, measureGroup, aiGroup]) {
    while (grp.children.length) {
      const c = grp.children.pop();
      c.traverse?.((o) => { o.geometry?.dispose?.(); });
    }
  }
  raycastTargets = [];
  savedMaterials.clear();
  contentGroup.matrixAutoUpdate = true;
  contentGroup.matrix.identity();
  contentGroup.position.set(0, 0, 0);
  contentGroup.quaternion.identity();
  floorMatrix = null;
  modelGroup.position.set(0, 0, 0);
  modelGroup.quaternion.identity();
  setScaleIdx(0);
}

function frameContent() {
  const box = new THREE.Box3().setFromObject(contentGroup);
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(size * 0.4, size * 0.3, size * 0.4));
  camera.near = Math.max(size / 1000, 0.01);
  camera.far = size * 20;
  camera.updateProjectionMatrix();
}

// ── in-scene AR HUD ──────────────────────────────────────────────────────────
// XRViewer (and other older XR browsers) do NOT support dom-overlay, so the
// HTML toolbar is invisible inside AR. This camera-anchored sprite toolbar is
// tappable through the XR select ray in ANY WebXR browser: tools, scale,
// clear, exit — plus a status line with capability diagnostics.

function hudCanvasTex(text, active, statusStyle = false) {
  const fs = statusStyle ? 30 : 44, pad = statusStyle ? 10 : 16;
  const cv = document.createElement('canvas');
  const cx = cv.getContext('2d');
  cx.font = `600 ${fs}px system-ui`;
  cv.width = Math.max(cx.measureText(text).width + pad * 2, statusStyle ? 40 : 100);
  cv.height = fs + pad * 2;
  cx.font = `600 ${fs}px system-ui`;
  cx.fillStyle = active ? 'rgba(46,160,67,0.95)' : 'rgba(24,29,36,0.92)';
  cx.fillRect(0, 0, cv.width, cv.height);
  if (!statusStyle) {
    cx.strokeStyle = active ? '#4ade80' : '#57606d';
    cx.lineWidth = 4;
    cx.strokeRect(2, 2, cv.width - 4, cv.height - 4);
  }
  cx.fillStyle = '#fff';
  cx.textAlign = 'center';
  cx.textBaseline = 'middle';
  cx.fillText(text, cv.width / 2, cv.height / 2);
  return { tex: new THREE.CanvasTexture(cv), aspect: cv.width / cv.height };
}

function makeHudSprite(action, text, h = 0.055, statusStyle = false) {
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ depthTest: false }));
  sp.renderOrder = 999;
  sp.userData = { action, hudButton: !statusStyle };
  sp.userData.redraw = (label, active = false) => {
    const { tex, aspect } = hudCanvasTex(label, active, statusStyle);
    sp.material.map?.dispose();
    sp.material.map = tex;
    sp.material.needsUpdate = true;
    sp.scale.set(h * aspect, h, 1);
  };
  sp.userData.redraw(text);
  return sp;
}

function layoutHud() {
  const gap = 0.012;
  const total = hudButtons.reduce((a, b) => a + b.scale.x, 0)
    + gap * (hudButtons.length - 1);
  let x = -total / 2;
  for (const b of hudButtons) {
    b.position.set(x + b.scale.x / 2, 0, 0);
    x += b.scale.x + gap;
  }
  hudStatus.position.set(0, 0.062, 0);
}

function buildHud() {
  removeHud();
  hudGroup = new THREE.Group();
  hudButtons = [];
  const defs = [['move', 'Move'], ['dist', 'Dist'], ['angle', 'Ang'],
                ['vol', 'Vol'], ['scale', SCALES[scaleIdx][1]],
                ['clear', 'Clear'], ['exit', '✕']];
  for (const [action, text] of defs) {
    const b = makeHudSprite(action, text);
    hudButtons.push(b);
    hudGroup.add(b);
  }
  hudStatus = makeHudSprite('', '', 0.034, true);
  hudGroup.add(hudStatus);
  layoutHud();
  refreshHud();
  scene.add(hudGroup);
}

function removeHud() {
  if (hudGroup) scene.remove(hudGroup);
  hudGroup = null;
  hudButtons = [];
  hudStatus = null;
}

function refreshHud() {
  if (!hudGroup) return;
  for (const b of hudButtons) {
    if (b.userData.action === 'scale') b.userData.redraw(SCALES[scaleIdx][1]);
    else b.userData.redraw(hudLabel(b.userData.action),
                           b.userData.action === currentTool);
  }
  hudStatus.userData.redraw(
    `${currentTool} · ${SCALES[scaleIdx][1]} · blend:${xrBlendMode || '?'}`
    + `${hitTestSource ? ' · hit-test' : ''}`);
  layoutHud();
}

function hudLabel(action) {
  return { move: 'Move', dist: 'Dist', angle: 'Ang', vol: 'Vol',
           clear: 'Clear', exit: '✕' }[action] || action;
}

function updateHudPlacement() {
  if (!hudGroup || !renderer.xr.isPresenting) return;
  const cam = renderer.xr.getCamera();
  const q = cam.quaternion;
  const p = cam.position.clone()
    .addScaledVector(new THREE.Vector3(0, 0, -1).applyQuaternion(q), 0.62)
    .addScaledVector(new THREE.Vector3(0, -1, 0).applyQuaternion(q), 0.20);
  hudGroup.position.lerp(p, 0.35);
  hudGroup.quaternion.slerp(q, 0.35);
}

function doHudAction(action) {
  if (action === 'exit') { renderer.xr.getSession()?.end(); return; }
  if (action === 'scale') { setScaleIdx(scaleIdx + 1); refreshHud(); return; }
  if (action === 'clear') { clearMeasures(); return; }
  setTool(action);
  refreshHud();
}

// ── AR session ────────────────────────────────────────────────────────────────

async function setupARButton() {
  const btn = $('btn-ar');
  const ok = navigator.xr && await navigator.xr.isSessionSupported?.('immersive-ar')
    .catch(() => false);
  const ios = /iPhone|iPad|iPod/.test(navigator.userAgent);
  tele('xr-support', { immersiveAR: !!ok, ios });
  btn.style.display = (ok || ios) ? 'block' : 'none';
  if (ok) {
    // native WebXR (Android Chrome and friends) — the canonical path
    btn.onclick = enterAR;
  } else if (ios) {
    // iOS Safari has NO WebXR (2026) — dedicated page on the self-hosted
    // 8th Wall engine (own SLAM + absolute metric scale over the camera)
    btn.onclick = () => {
      if (!session?.has_mesh) { toast('XR needs a session with a mesh'); return; }
      location.href = `ios.html?session=${encodeURIComponent(session.id)}`;
    };
  } else {
    toast('WebXR AR not available in this browser — 3D mode only', 4000);
  }
}

async function enterAR() {
  try {
    const xrSession = await navigator.xr.requestSession('immersive-ar', {
      requiredFeatures: [],
      optionalFeatures: ['local-floor', 'hit-test', 'dom-overlay'],
      domOverlay: { root: $('overlay') },
    });
    // older XR browsers (Mozilla XRViewer) may not grant local-floor
    let refType = 'local-floor';
    try { await xrSession.requestReferenceSpace('local-floor'); }
    catch { refType = 'local'; }
    renderer.xr.setReferenceSpaceType(refType);
    reticle.matrixAutoUpdate = false;          // XR drives it via matrix
    reticle.scale.setScalar(1);
    xrRefType = refType;
    await renderer.xr.setSession(xrSession);
    xrRefSpace = renderer.xr.getReferenceSpace();
    xrBlendMode = xrSession.environmentBlendMode || '?';
    try {
      const viewerSpace = await xrSession.requestReferenceSpace('viewer');
      hitTestSource = await xrSession.requestHitTestSource?.({ space: viewerSpace }) || null;
    } catch { hitTestSource = null; }
    // AR entry: NOTHING placed yet — auto-placing a building-sized model
    // engulfed the user (its walls occluded the camera AND the reticle, which
    // read as "no passthrough"). Camera + reticle only; the user aims and
    // taps to place a 1:10 miniature on the real floor, then scales up.
    contentGroup.visible = false;
    lastAnchor = null;
    setScaleIdx(1);                            // 1:10 tabletop first
    modelGroup.position.set(0, 0, 0);
    buildHud();
    const hadOverlay = !!xrSession.domOverlayState;
    tele('ar-start', {
      blend: xrBlendMode, refType, hitTest: !!hitTestSource,
      domOverlay: hadOverlay, session: session?.id,
      interactionMode: xrSession.interactionMode || null,
    });
    xrSession.addEventListener('select', onXRSelect);
    xrSession.addEventListener('end', () => {
      const diag = `AR caps — blend:${xrBlendMode} · hit-test:${!!hitTestSource}`
        + ` · dom-overlay:${hadOverlay} · ref:${xrRefType}`;
      hitTestSource = null;
      reticle.visible = false;
      removeHud();
      contentGroup.visible = true;
      lastAnchor = null;
      setScaleIdx(0);
      modelGroup.position.set(0, 0, 0);
      frameContent();
      toast(diag, 12000);       // report what the browser actually supported
    });
    toast('Tap to place (Move) — switch tools below');
  } catch (e) {
    tele('ar-error', { msg: String(e.message || e) });
    toast(`AR failed: ${e.message || e}`, 5000);
  }
}

function onXRSelect(ev) {
  const frame = ev.frame;
  // ray from the input source (screen tap on handheld AR)
  const pose = frame.getPose(ev.inputSource.targetRaySpace, xrRefSpace);
  let origin = null, dir = null;
  if (pose) {
    const m = new THREE.Matrix4().fromArray(pose.transform.matrix);
    origin = new THREE.Vector3().setFromMatrixPosition(m);
    dir = new THREE.Vector3(0, 0, -1).applyMatrix4(
      new THREE.Matrix4().extractRotation(m)).normalize();
    // the in-scene HUD has tap priority over the model
    raycaster.set(origin, dir);
    raycaster.camera = renderer.xr.getCamera();   // Sprite.raycast needs it
    const hudHit = raycaster.intersectObjects(hudButtons, false);
    if (hudHit.length) { doHudAction(hudHit[0].object.userData.action); return; }
  }
  if (currentTool === 'move') {
    if (reticle.visible) {
      const p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
      reticle.matrix.decompose(p, q, s);
      placeContentAt(p.x, p.z, p.y);           // bbox on the REAL tapped floor
    } else {
      // no hit-test: drop 2 m in front of the camera at floor-guess height
      const cam = renderer.xr.getCamera();
      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
      const y = xrRefType === 'local' ? cam.position.y - 1.4 : 0;
      const t = cam.position.clone().addScaledVector(fwd, 2.0);
      placeContentAt(t.x, t.z, y);
    }
    return;
  }
  if (origin && dir) pickAndMeasure(origin, dir);
}

// ── picking (shared XR / non-XR) ─────────────────────────────────────────────

const raycaster = new THREE.Raycaster();
raycaster.firstHitOnly = true;        // three-mesh-bvh fast path

function pickAndMeasure(origin, dir) {
  raycaster.set(origin, dir);
  raycaster.params.Points.threshold = 0.02 * displayScale * 3;
  const hits = raycaster.intersectObjects(raycastTargets, false);
  if (!hits.length) { toast('No surface under the tap'); return; }
  const world = hits[0].point.clone();
  const model = modelGroup.worldToLocal(world.clone());   // METRIC coordinates
  addMeasurePoint(model);
}

function bindTapPicking() {
  let downPos = null;
  renderer.domElement.addEventListener('pointerdown', (e) => {
    downPos = [e.clientX, e.clientY];
  });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (renderer.xr.isPresenting) return;              // XR taps come via 'select'
    if (!downPos) return;
    const dx = e.clientX - downPos[0], dy = e.clientY - downPos[1];
    downPos = null;
    if (Math.hypot(dx, dy) > 8) return;                             // drag = orbit
    const ndc = new THREE.Vector2((e.clientX / innerWidth) * 2 - 1,
                                  -(e.clientY / innerHeight) * 2 + 1);
    if (currentTool === 'move') {
      if (camAR.on) {
        // place at the reticle (screen center) — tap anywhere confirms
        if (reticle.visible) placeContentAt(reticle.position.x, reticle.position.z);
        else camARPlace(ndc);
        toast('Placed — switch to Dist/Ang/Vol to measure');
      }
      return;
    }
    raycaster.setFromCamera(ndc, camera);
    raycaster.params.Points.threshold = 0.02 * displayScale * 3;
    const hits = raycaster.intersectObjects(raycastTargets, false);
    if (!hits.length) return;
    addMeasurePoint(modelGroup.worldToLocal(hits[0].point.clone()));
  });
}

// ── measurement tools (all math in metric MODEL space) ───────────────────────

let pending = [];        // clicked model-space points for the current tool

function setTool(tool) {
  currentTool = tool;
  pending = [];
  document.querySelectorAll('#toolbar [data-tool]').forEach((b) =>
    b.classList.toggle('active', b.dataset.tool === tool));
  const hints = {
    move: 'Move: tap the floor to place the model (AR)',
    dist: 'Distance: tap two points',
    angle: 'Angle: tap 3 points (vertex second)',
    vol: 'Volume: tap 2 base corners, then a height point',
  };
  toast(hints[tool] || tool);
}

function fmt(m) {
  return m >= 1 ? `${m.toFixed(2)} m` : `${(m * 100).toFixed(1)} cm`;
}

function marker(p, color = 0xffc107) {
  const s = new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 12),
                           new THREE.MeshBasicMaterial({ color }));
  s.position.copy(p);
  measureGroup.add(s);
  return s;
}

function line(points, color = 0xffc107) {
  const g = new THREE.BufferGeometry().setFromPoints(points);
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color }));
  measureGroup.add(l);
  return l;
}

function label(text, p, group = measureGroup) {
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
  const tex = new THREE.CanvasTexture(cv);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
  const h = 0.12;                                    // 12 cm tall at 1:1
  sp.scale.set(h * cv.width / cv.height, h, 1);
  sp.position.copy(p);
  group.add(sp);
  return sp;
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
      const deg = THREE.MathUtils.radToDeg(Math.acos(
        THREE.MathUtils.clamp(u1.dot(u2), -1, 1)));
      label(`${deg.toFixed(1)}°`, v.clone().addScaledVector(
        u1.clone().add(u2).normalize(), 0.25));
      pending = [];
    }
  } else if (currentTool === 'vol') {
    pending.push(p); marker(p, 0x4ade80);
    if (pending.length === 3) {
      const [a, b, c] = pending;
      const y0 = Math.min(a.y, b.y);
      const h = Math.max(Math.abs(c.y - y0), 0.05);
      const min = new THREE.Vector3(Math.min(a.x, b.x), y0, Math.min(a.z, b.z));
      const max = new THREE.Vector3(Math.max(a.x, b.x), y0 + h, Math.max(a.z, b.z));
      const size = max.clone().sub(min);
      const vol = size.x * size.y * size.z;
      const geo = new THREE.BoxGeometry(size.x, size.y, size.z);
      const box = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
        color: 0x4ade80, transparent: true, opacity: 0.18, depthWrite: false }));
      box.position.copy(min).add(size.clone().multiplyScalar(0.5));
      measureGroup.add(box);
      measureGroup.add(new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0x4ade80 }))).children.at(-1)
        .position.copy(box.position);
      label(`${size.x.toFixed(2)}×${size.z.toFixed(2)}×${size.y.toFixed(2)} m = ${vol.toFixed(2)} m³`,
            box.position.clone().setY(max.y + 0.1));
      pending = [];
    }
  }
}

// ── AI chat (Phase 5 spatial QA) ─────────────────────────────────────────────

function chatMsg(html, cls = '') {
  const d = document.createElement('div');
  d.className = `msg ${cls}`;
  d.innerHTML = html;
  $('msgs').appendChild(d);
  $('msgs').scrollTop = $('msgs').scrollHeight;
  return d;
}

function esc(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

// best-effort: pull [x,y,z] positions out of tool results and mark them in 3D
function extractPoints(obj, out = [], depth = 0) {
  if (depth > 6 || out.length > 20 || obj == null) return out;
  if (Array.isArray(obj)) {
    if (obj.length === 3 && obj.every((v) => typeof v === 'number')) {
      out.push(new THREE.Vector3(obj[0], obj[1], obj[2]));
    } else obj.forEach((v) => extractPoints(v, out, depth + 1));
  } else if (typeof obj === 'object') {
    for (const [k, v] of Object.entries(obj)) {
      if (/pos|point|center|centroid|p1|p2|start|end|corner/i.test(k)) {
        extractPoints(v, out, depth + 1);
      } else if (typeof v === 'object') extractPoints(v, out, depth + 1);
    }
  }
  return out;
}

async function askAI(question) {
  chatMsg(esc(question), 'q');
  const wait = chatMsg('…thinking (deterministic tools measure, the VLM narrates)');
  try {
    const r = await fetch(`${API}/api/spatial_qa`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: session.id }),
    });
    const data = await r.json();
    if (r.status === 503 && data.status === 'loading') {
      wait.innerHTML = '⏳ The VLM is loading on the pod GPU (it frees VRAM during '
        + 'reconstruction) — ask again in ~2 minutes.';
      return;
    }
    if (!r.ok) { wait.innerHTML = `⚠ ${esc(data.error || r.status)}`; return; }
    let html = esc(data.answer || '(no answer)');
    if (data.tool_trace?.length) {
      const tr = data.tool_trace.map((t) =>
        `▸ ${esc(t.tool || t.name || '?')}(${esc(JSON.stringify(t.args ?? t.arguments ?? {}))})`
      ).join('\n');
      html += `<div class="trace">${tr}</div>`;
    }
    wait.innerHTML = html;
    // markers from trace results
    while (aiGroup.children.length) aiGroup.children.pop();
    // store coordinates live in the RAW model frame — upright them like the content
    const pts = extractPoints(data.tool_trace)
      .map((p) => (floorMatrix ? p.applyMatrix4(floorMatrix) : p));
    pts.slice(0, 20).forEach((p, i) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 12),
                               new THREE.MeshBasicMaterial({ color: 0xd65db1 }));
      m.position.copy(p);
      aiGroup.add(m);
      if (i < 8) label(`✦${i + 1}`, p.clone().add(new THREE.Vector3(0, 0.12, 0)), aiGroup);
    });
    if (pts.length) toast(`${pts.length} AI reference point(s) marked in the scene`);
  } catch (e) {
    wait.innerHTML = `⚠ ${esc(e.message || e)}`;
  }
}

// ── Cam AR: camera-feed + gyro fallback (no WebXR needed — works in Safari) ──
// WebXR Viewer reports blend:opaque (it never composites the camera), so real
// passthrough needs a browser-agnostic path: getUserMedia environment camera
// as a fullscreen <video> behind the transparent WebGL canvas, and
// deviceorientation driving the virtual camera (permission flow + YXZ euler +
// screen-orientation compensation ported from the legacy static/xr_viewer.html).
// No positional tracking: the user pans from a standpoint and re-places by
// tapping the floor. Pinch calibrates the FOV against real references so 1:1
// reads correctly on screen.

let camAR = { on: false, stream: null, yawPitch: null, initialAlpha: null,
              screenO: 0, baseFov: 62 };

// Canonical device-orientation → camera quaternion, verbatim from three.js's
// battle-tested DeviceOrientationControls: YXZ euler, then the fixed -90° X
// correction (the camera looks out the BACK of the device, not the top), then
// the live screen-orientation twist. The earlier ad-hoc `beta − π/2` version
// coupled the axes — panning the phone rotated the world.
const _zee = new THREE.Vector3(0, 0, 1);
const _qBack = new THREE.Quaternion(-Math.SQRT1_2, 0, 0, Math.SQRT1_2);

function camQuatFromOrientation(ev) {
  const alpha = THREE.MathUtils.degToRad(ev.alpha || 0);
  const beta = THREE.MathUtils.degToRad(ev.beta || 0);
  const gamma = THREE.MathUtils.degToRad(ev.gamma || 0);
  if (camAR.initialAlpha === null) camAR.initialAlpha = alpha;   // yaw zero = where you look on enable
  const orient = THREE.MathUtils.degToRad(
    (screen.orientation?.angle ?? window.orientation) || 0);
  const q = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(beta, alpha - camAR.initialAlpha, -gamma, 'YXZ'));
  q.multiply(_qBack);
  q.multiply(new THREE.Quaternion().setFromAxisAngle(_zee, -orient));
  return q;
}

function onDeviceOrientation(ev) {
  if (!camAR.on) return;
  camera.quaternion.copy(camQuatFromOrientation(ev));
}

async function enterCamAR() {
  if (camAR.on) { exitCamAR(); return; }
  try {
    // iOS 13+ requires an explicit user-gesture permission for the gyro
    if (typeof DeviceOrientationEvent !== 'undefined'
        && typeof DeviceOrientationEvent.requestPermission === 'function') {
      const p = await DeviceOrientationEvent.requestPermission();
      if (p !== 'granted') { toast('Motion permission denied'); return; }
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' }, audio: false });
    const v = $('camfeed');
    v.srcObject = stream;
    v.style.display = 'block';
    await v.play();
    camAR.stream = stream;
    camAR.on = true;
    camAR.initialAlpha = null;
    controls.enabled = false;
    camera.position.set(0, 1.5, 0);            // standing eye height
    camera.fov = camAR.baseFov;
    camera.updateProjectionMatrix();
    setScaleIdx(0);                            // metric first
    modelGroup.position.set(0, 0, 0);
    placeContentAt(0, -6);                     // content bbox 6 m ahead, floor on ground
    window.addEventListener('deviceorientation', onDeviceOrientation);
    setTool('move');
    $('btn-camar').classList.add('active');
    tele('camar-start', { fov: camAR.baseFov });
    toast('Cam AR: pan the phone · tap the floor to re-place · pinch = FOV calibration');
    navigator.wakeLock?.request('screen').catch(() => {});   // optional
  } catch (e) {
    tele('camar-error', { msg: String(e.message || e) });
    toast(`Cam AR failed: ${e.message || e}`, 5000);
  }
}

function exitCamAR() {
  camAR.on = false;
  reticle.visible = false;
  reticle.scale.setScalar(1);
  camAR.stream?.getTracks().forEach((t) => t.stop());
  camAR.stream = null;
  const v = $('camfeed');
  v.srcObject = null;
  v.style.display = 'none';
  window.removeEventListener('deviceorientation', onDeviceOrientation);
  $('btn-camar').classList.remove('active');
  controls.enabled = true;
  camera.fov = 60;
  camera.updateProjectionMatrix();
  modelGroup.position.set(0, 0, 0);
  frameContent();
}

// Place the CONTENT at a world point. The scene's own origin is wherever the
// capture walk started — often tens of meters away from the geometry — so
// placing the group origin makes the building land far away and read tiny.
// This puts the content's bbox center on the target and its bbox floor at y=0.
let lastAnchor = null;   // last placed world point — scale cycles re-anchor here

function placeContentAt(x, z, y = 0) {
  contentGroup.visible = true;
  const box = new THREE.Box3().setFromObject(contentGroup);
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3());
  modelGroup.position.x += x - c.x;
  modelGroup.position.z += z - c.z;
  // The pipeline already calibrated the model's floor at y=0 (baked orient or
  // floor_transform) — pin THAT plane to the tapped ground height. Never use
  // bbox.min.y: below-floor noise points made the model float by that margin.
  modelGroup.position.y = y;
  lastAnchor = new THREE.Vector3(x, y, z);
}

// Move tool in Cam AR: tap → ray → virtual floor plane y=0 → re-place there
const _floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
function camARPlace(ndc) {
  raycaster.setFromCamera(ndc, camera);
  const hit = new THREE.Vector3();
  if (raycaster.ray.intersectPlane(_floorPlane, hit)) {
    placeContentAt(hit.x, hit.z);
  }
}

// pinch = FOV calibration (match the virtual FOV to the phone camera's so 1:1
// reads true on screen); metric scale itself is never touched
let _pinchD = null;
function bindPinchFov() {
  const el = renderer.domElement;
  el.addEventListener('touchmove', (e) => {
    if (!camAR.on || e.touches.length !== 2) { _pinchD = null; return; }
    e.preventDefault();
    const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX,
                         e.touches[0].clientY - e.touches[1].clientY);
    if (_pinchD !== null) {
      camera.fov = THREE.MathUtils.clamp(camera.fov * (_pinchD / d), 35, 85);
      camAR.baseFov = camera.fov;
      camera.updateProjectionMatrix();
      toast(`FOV ${camera.fov.toFixed(0)}°`, 800);
    }
    _pinchD = d;
  }, { passive: false });
  el.addEventListener('touchend', () => { _pinchD = null; });
}

// ── AR Quick Look (iOS native): USDZ prepared on the SERVER ──────────────────
// Apple's own AR viewer: perfect passthrough + ARKit anchoring at true metric
// scale (USDZ meters). The pod decimates + packages the textured mesh
// (UV-preserving, ARKit-compliant, cached); the phone only downloads and
// opens it — on-device conversion OOM-crashed the tab on 4M-tri scenes.

const _sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function quickLook() {
  if (!session?.has_mesh) {
    toast('Quick Look needs a session with a mesh');
    return;
  }
  loading(true, 'Preparing USDZ on the server… (first time ~2-3 min, then cached)');
  try {
    for (let i = 0; i < 90; i++) {                    // poll up to ~7.5 min
      const r = await fetch(
        `${API}/api/ar/usdz/${encodeURIComponent(session.id)}?prepare=1`);
      const st = await r.json();
      if (!r.ok) throw new Error(st.error || `HTTP ${r.status}`);
      if (st.status === 'ready') break;
      loading(true, `Preparing USDZ… (server is decimating + packaging)`);
      await _sleep(5000);
    }
    const a = document.createElement('a');
    a.rel = 'ar';
    a.href = `${API}/api/ar/usdz/${encodeURIComponent(session.id)}`;
    a.appendChild(document.createElement('img'));     // Safari requires an <img> child
    document.body.appendChild(a);
    a.click();
    a.remove();
    tele('quicklook-open', { session: session.id });
  } catch (e) {
    tele('quicklook-error', { msg: String(e.message || e) });
    toast(`Quick Look failed: ${e.message || e}`, 6000);
  } finally {
    loading(false);
  }
}

// ── scale / lighting / wiring ────────────────────────────────────────────────

function clearMeasures() {
  pending = [];
  while (measureGroup.children.length) measureGroup.children.pop();
  while (aiGroup.children.length) aiGroup.children.pop();
}

function setScaleIdx(i) {
  scaleIdx = i % SCALES.length;
  displayScale = SCALES[scaleIdx][0];
  modelGroup.scale.setScalar(displayScale);
  $('btn-scale').textContent = SCALES[scaleIdx][1];
  // keep the placed anchor fixed while cycling scales (otherwise scaling
  // swings the content away from the tapped point)
  if (lastAnchor && renderer?.xr.isPresenting) {
    placeContentAt(lastAnchor.x, lastAnchor.z, lastAnchor.y);
  }
}

function wireUI() {
  $('btn-back').onclick = () => {
    renderer?.xr.getSession()?.end();
    if (camAR.on) exitCamAR();
    $('viewer').style.display = 'none';
    $('home').style.display = 'block';
  };
  const _camarBtn = $('btn-camar');            // retired from the UI; code kept
  if (_camarBtn) _camarBtn.onclick = () => enterCamAR();
  const isIOS = /iPhone|iPad|iPod/.test(navigator.userAgent);
  $('btn-ql').style.display = isIOS ? 'block' : 'none';
  $('btn-ql').onclick = () => quickLook();
  $('btn-scale').onclick = () => setScaleIdx(scaleIdx + 1);
  $('btn-light').onclick = () => { fullbright = !fullbright; applyLighting(); };
  $('btn-clear').onclick = clearMeasures;
  document.querySelectorAll('#toolbar [data-tool]').forEach((b) => {
    b.onclick = () => setTool(b.dataset.tool);
  });
  $('btn-chat').onclick = () => { $('chat').style.display = 'flex'; };
  $('btn-chat-close').onclick = () => { $('chat').style.display = 'none'; };
  $('chatform').onsubmit = (e) => {
    e.preventDefault();
    const q = $('chat-in').value.trim();
    if (!q) return;
    $('chat-in').value = '';
    if (!session?.has_ai) {
      chatMsg('⚠ This session has no instance store (scene_r.db) — run the '
        + 'segmentation + store build first, then the AI can measure it.', '');
      return;
    }
    askAI(q);
  };
}

wireUI();
loadSessions();
