#!/usr/bin/env node
/**
 * mock-server.mjs — minimal backend for the screenshot harness (prompt_ui.txt
 * §11). Serves recorded-shape responses for every endpoint the views call,
 * a synthetic Potree 2.0 octree (a room: floor, two walls, a column) with
 * rgb / classification / confidence / status / mv_votes attributes, and the
 * three WebSocket channels (/ws/viewer, /ws/logs, /ws/team) with a
 * hand-rolled RFC 6455 server (no extra dependency).
 *
 * Scenarios switch the fixture set: POST /__mock/scenario {"name": "..."}
 *   empty        — connected, no sessions
 *   sessions     — sessions, none loaded
 *   cloud        — a session with a cloud, 3 instances, no acta
 *   pipeline     — a pipeline running on the session (stages + banner)
 *   certify      — cloud + certification acta with pending epochs, edges,
 *                  duplicates, attention items, witness fields
 *   correction   — cloud + a pending correction awaiting the verdict
 *   bim          — cloud + BIM comparison results (deviation report)
 * Usage: node scripts/mock-server.mjs [port=8790]
 */
import http from 'node:http'
import crypto from 'node:crypto'

const PORT = Number(process.argv[2] || 8790)
let scenario = 'sessions'

// ── fixtures ────────────────────────────────────────────────────────────
const USER = { id: 1, username: 'hbarreto', email: 'h@ingerop.mx', role: 'admin', full_name: 'Hernán Barreto', avatar_url: null, is_active: true, created_at: '2026-01-10T10:00:00Z', last_login: '2026-09-14T08:30:00Z' }
const USERS = [USER,
  { id: 2, username: 'topografia', email: null, role: 'editor', full_name: 'Topografía Norte', avatar_url: null, is_active: true, created_at: null, last_login: '2026-09-12T15:00:00Z' },
  { id: 3, username: 'supervisor', email: 'sup@ingerop.mx', role: 'manager', full_name: 'Supervisión de obra', avatar_url: null, is_active: false, created_at: null, last_login: null }]
const SESSIONS = [
  { id: 'tunel-norte', frame_count: 4820, has_cloud: true, has_segments: true, has_bim: false, bim_count: 0, cloud_size_mb: 812, has_sabana: false, date: '2026-09-02' },
  { id: 'estacion-sur', frame_count: 2210, has_cloud: true, has_segments: false, has_bim: true, bim_count: 1, cloud_size_mb: 340, has_sabana: true, date: '2026-08-21' },
  { id: 'patio-taller', frame_count: 0, has_cloud: false, has_segments: false, has_bim: false, bim_count: 0, cloud_size_mb: 0, has_sabana: false, date: '2026-09-13' },
]
const SCANS = {
  'tunel-norte': [
    { key: '2026-09-02/default', label: 'recorrido base', date: '2026-09-02', kind: 'scan', is_reference: true, is_active: true, points: 18_400_000, has_potree: true, recon_state: 'complete', frame_count: 2400, source: 'default', has_output: true },
    { key: '2026-09-10/default', label: 'avance semana 3', date: '2026-09-10', kind: 'scan', is_reference: false, is_active: false, points: 16_900_000, has_potree: true, recon_state: 'complete', frame_count: 2420, source: 'default', has_output: true },
    { key: 'merged/2026-09-10', label: 'composición', date: '2026-09-10', kind: 'fused', is_reference: false, is_active: false, points: 35_100_000, has_potree: true },
  ],
  'estacion-sur': [{ key: '2026-08-21/default', label: 'levantamiento', date: '2026-08-21', kind: 'scan', is_reference: true, is_active: true, points: 7_800_000, has_potree: true, recon_state: 'partial', cached_chunks: 3, frame_count: 2210, source: 'default', has_output: true }],
  'patio-taller': [],
}
const INSTANCES = [
  { id: 1, instance_id: 1, global_id: 'wall_1', label: 'wall', color: '#1285ed', total_points: 2_140_000, excluded: false, obb: { center: [0, 1.5, -3], half_extents: [4, 1.5, 0.15], rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] } },
  { id: 2, instance_id: 2, global_id: 'column_2', label: 'column', color: '#15afa4', total_points: 380_000, excluded: false, obb: { center: [2, 1.5, 0], half_extents: [0.3, 1.5, 0.3], rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] } },
  { id: 3, instance_id: 3, global_id: 'floor_3', label: 'floor', color: '#a08b67', total_points: 5_600_000, excluded: false, obb: { center: [0, 0, 0], half_extents: [4, 0.05, 3], rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] } },
]
const PIPELINE = { session_id: 'tunel-norte', status: 'running', current_stage_idx: 1, stages: [
  { id: 'recon', label: 'Reconstrucción', icon: '', enabled: true, status: 'done', pct: 100, message: 'nube métrica lista', elapsed: 4120 },
  { id: 'vlm', label: 'VLM', icon: '', enabled: true, status: 'running', pct: 42, message: 'describiendo 214 / 512 keyframes', elapsed: 380 },
  { id: 'sam3', label: 'SAM3', icon: '', enabled: true, status: 'pending', pct: 0, message: '', elapsed: 0 },
  { id: 'cloudcompy', label: 'CloudComPy', icon: '', enabled: true, status: 'pending', pct: 0, message: '', elapsed: 0 },
  { id: 'certify', label: 'Certificación', icon: '', enabled: true, status: 'pending', pct: 0, message: '', elapsed: 0 },
] }
const ACTA = { stop_reason: 'converged', elapsed_s: 612, iterations: [
  { iteration: 1, verdict: 'applied', objective_prev: 0.318, objective: 0.142, gate_mode: 'advisory', gates: [
    { name: 'drift_budget', passed: true, advisory: true, value: 0.031, threshold: 0.05, detail: '3.1 cm / 100 m' },
    { name: 'loop_gain', passed: true, advisory: true, value: 0.55, threshold: 0.2 },
    { name: 'held_out_pairs', passed: false, advisory: true, value: 0.024, threshold: 0.02, detail: 'p95 2.4 cm' },
  ] },
  { iteration: 2, verdict: 'stopped', objective_prev: 0.142, objective: 0.139, gate_mode: 'advisory', gates: [{ name: 'min_loop_gain', passed: false, advisory: true, value: 0.02, threshold: 0.05 }] },
] }
const EDGES = {
  positions: [[0, 1.5, 4], [3, 1.5, 2], [3, 1.5, -2], [0, 1.5, -4], [-3, 1.5, -2], [-3, 1.5, 2], [0.2, 1.5, 3.8]],
  odometry: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
  loops: [
    { i: 0, j: 6, kind: 'accepted', source: 'salad', residual_m: 0.018, observability: 'full' },
    { i: 1, j: 5, kind: 'scale_break', source: 'instance', residual_m: 0.061, reason: 'chunk scale differs 1.8 %' },
    { i: 2, j: 4, kind: 'rejected', source: 'visit', residual_m: 0.31, reason: 'ICP did not converge' },
  ],
  duplicates: [{ label: 'column', instance_id: 2, i: 1, j: 5, separation_m: 0.084, closure_after_m: 0.011, verdict: 'closed', i_pos: [2, 1.5, 0], j_pos: [2.08, 1.5, 0.02] }],
}
const ATTENTION = { n: 3, items: [
  { kind: 'scale_break', severity: 3, text: 'Ruptura de escala entre los chunks 2 y 3: 1.8 % de diferencia, se aplicó la escala relativa.', anchor: { position: [3, 1.5, 2] }, anchor_b: { position: [-3, 1.5, 2] } },
  { kind: 'rejected_loop', severity: 2, text: 'Cierre rechazado entre kf 2 y kf 4: ICP no convergió; revisar visualmente.', anchor: { position: [3, 1.5, -2] } },
  { kind: 'held_out', severity: 1, text: 'Pares reservados con p95 2.4 cm, apenas sobre el umbral consultivo de 2 cm.' },
] }
const REPORT = { epoch: 2, available_epochs: [0, 1, 2], comparison_vs_previous: { objective: -0.176 }, metrics: {
  objective: 0.139, loop_residual: { median_demand_m: 0.021, n_accepted: 2, n_edges: 3 }, seam_residual: { median_m: 0.008, p95_m: 0.024 }, closure: { median_m: 0.014, max_m: 0.031 },
  duplicates: { n: 0, target: 0 }, scale: { applied: true, r_per_chunk: [1.0, 1.0182, 1.0], scale_break: true },
  witnesses: { status_fraction: { verified: 0.71, single_witness: 0.19, mask_conflict: 0.03, dynamic: 0.02, unobserved: 0.05 }, mv_votes_mean: 3.4, mask_conflicts_total: 1240, status_counts: {} },
  depth_disagreement: { holdout_rel_after: 0.012, applied: true }, known_dimensions: null, loop_coverage: { coverage: 0.64 }, authority: { pose_graph: { fraction_used: 0.38, saturated: false } },
} }
const SABANA_META = { date: '2026-09-11T09:12:00Z', tolerance_mm: 15, total_points: 7_800_000, global_advance_pct: 62, summary: { total_elements: 6, evaluated: 4, unmatched: 2, errors: 0 }, elements: [
  { element_key: 'W1', label: 'Muro andén:W1', ifc_type: 'IfcWall', status: 'evaluated', quality: 'good', advance_pct: 96, coverage_cumulative: 96, occluded_pct: 0, element_state: 'COMPLETED', correctness_pct: 97.2, mean_mm: 6.1, bim_surface_m2: 42.5 },
  { element_key: 'C3', label: 'Columna:C3', ifc_type: 'IfcColumn', status: 'evaluated', quality: 'regular', advance_pct: 78, coverage_cumulative: 81, occluded_pct: 12, element_state: 'IN_PROGRESS', correctness_pct: 84.0, mean_mm: 12.8, bim_surface_m2: 6.2 },
  { element_key: 'S2', label: 'Losa:S2', ifc_type: 'IfcSlab', status: 'evaluated', quality: 'bad', advance_pct: 40, coverage_cumulative: 40, occluded_pct: 30, element_state: 'OCCLUDED_FROZEN', correctness_pct: 61.5, mean_mm: 24.3, bim_surface_m2: 88.0 },
  { element_key: 'B4', label: 'Viga:B4', ifc_type: 'IfcBeam', status: 'evaluated', quality: 'good', advance_pct: 100, coverage_cumulative: 100, occluded_pct: 0, element_state: 'VERIFIED', correctness_pct: 99.1, mean_mm: 3.2, bim_surface_m2: 4.4 },
  { element_key: 'D5', label: 'Puerta:D5', ifc_type: 'IfcDoor', status: 'unmatched' },
  { element_key: 'W6', label: 'Muro:W6', ifc_type: 'IfcWall', status: 'unmatched' },
] }
const COMPARE = { ok: true, tolerance_mm: 15, results: SABANA_META.elements.filter(e => e.status === 'evaluated').map((e, i) => ({
  element_key: e.element_key, label: e.label, status: 'evaluated', ifc_type: e.ifc_type,
  stats: { min_mm: 0, max_mm: [22, 41, 78, 14][i], mean_mm: e.mean_mm, std_mm: 4, median_mm: e.mean_mm, p95_mm: [14, 28, 55, 9][i], within_tolerance: [1_900_000, 620_000, 410_000, 300_000][i], total_points: [2_000_000, 800_000, 1_100_000, 305_000][i], pass_rate: [95, 77, 37, 98][i], tolerance_mm: 15 },
  histogram: { counts: [1200, 900, 700, 500, 320, 180, 90, 40, 12, 4], bin_edges_mm: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] },
})) }
const TEAMS = [{ id: 1, name: 'Tramo norte', description: 'Supervisión túnel', manager_id: 1, manager_name: 'Hernán Barreto', is_active: true,
  members: USERS.map(u => ({ user_id: u.id, username: u.username, full_name: u.full_name, role: u.role, avatar_url: null })), sessions: [{ id: 1, session_id: 'tunel-norte' }, { id: 2, session_id: 'estacion-sur' }] }]
const MESSAGES = [{ id: 1, team_id: 1, user_id: 2, username: 'topografia', content: 'Subí el recorrido de la semana 3, ya está reconstruyendo.', timestamp: '2026-09-14T08:12:00Z' }, { id: 2, team_id: 1, user_id: 1, username: 'hbarreto', content: 'Perfecto, reviso la certificación cuando termine.', timestamp: '2026-09-14T08:15:00Z' }]
const ACTIVITY = [{ id: 1, user_id: 2, username: 'topografia', action: 'pipeline_started', detail: 'tunel-norte 2026-09-10', timestamp: '2026-09-14T08:10:00Z' }, { id: 2, user_id: 1, username: 'hbarreto', action: 'session_loaded', detail: 'tunel-norte', timestamp: '2026-09-14T08:14:00Z' }]

const hasCloud = () => ['cloud', 'pipeline', 'certify', 'correction', 'bim'].includes(scenario)
const sessions = () => scenario === 'empty' ? [] : SESSIONS

// ── synthetic Potree 2.0 octree (single leaf node) ──────────────────────
function buildOctree() {
  const pts = []
  const rnd = mulberry32(7)
  const push = (x, y, z, r, g, b, cls, status) => pts.push({ x, y, z, r, g, b, cls, conf: 0.5 + rnd() * 0.5, status, votes: 1 + Math.floor(rnd() * 6) })
  for (let i = 0; i < 26000; i++) { const x = -4 + rnd() * 8, z = -3 + rnd() * 6; push(x, (rnd() - 0.5) * 0.02, z, 110 + rnd() * 20, 105 + rnd() * 20, 95 + rnd() * 20, 3, rnd() < 0.8 ? 0 : 1) }
  for (let i = 0; i < 12000; i++) { const x = -4 + rnd() * 8, y = rnd() * 3; push(x, y, -3 + (rnd() - 0.5) * 0.03, 160 + rnd() * 30, 150 + rnd() * 30, 140 + rnd() * 30, 1, rnd() < 0.7 ? 0 : rnd() < 0.5 ? 1 : 2) }
  for (let i = 0; i < 6000; i++) { const z = -3 + rnd() * 6, y = rnd() * 3; push(-4 + (rnd() - 0.5) * 0.03, y, z, 150 + rnd() * 30, 140 + rnd() * 30, 130 + rnd() * 30, 0, 0) }
  for (let i = 0; i < 4000; i++) { const a = rnd() * Math.PI * 2, y = rnd() * 3; push(2 + 0.3 * Math.cos(a), y, 0.3 * Math.sin(a), 120 + rnd() * 20, 120 + rnd() * 20, 125 + rnd() * 20, 2, rnd() < 0.9 ? 0 : 3) }
  for (let i = 0; i < 800; i++) push(-2 + rnd() * 0.6, 0.4 + rnd() * 0.6, 1 + rnd() * 0.6, 200, 60, 40, 0, 4)
  const min = [-4.1, -0.05, -3.1], max = [4.1, 3.05, 3.1]
  const scale = 0.001
  const stride = 12 + 6 + 1 + 4 + 1 + 1
  const buf = Buffer.alloc(pts.length * stride)
  pts.forEach((p, i) => {
    const o = i * stride
    buf.writeInt32LE(Math.round((p.x - min[0]) / scale), o); buf.writeInt32LE(Math.round((p.y - min[1]) / scale), o + 4); buf.writeInt32LE(Math.round((p.z - min[2]) / scale), o + 8)
    buf.writeUInt16LE(Math.round(p.r * 257), o + 12); buf.writeUInt16LE(Math.round(p.g * 257), o + 14); buf.writeUInt16LE(Math.round(p.b * 257), o + 16)
    buf.writeUInt8(p.cls, o + 18); buf.writeFloatLE(p.conf, o + 19); buf.writeUInt8(p.status, o + 23); buf.writeUInt8(p.votes, o + 24)
  })
  const hier = Buffer.alloc(22)
  hier.writeUInt8(1, 0); hier.writeUInt8(0, 1); hier.writeUInt32LE(pts.length, 2); hier.writeBigUInt64LE(0n, 6); hier.writeBigUInt64LE(BigInt(buf.length), 14)
  const metadata = {
    version: '2.0', name: 'mock', description: '', points: pts.length, projection: '', hierarchy: { firstChunkSize: 22, stepSize: 4, depth: 0 },
    offset: min, scale: [scale, scale, scale], spacing: 0.05, boundingBox: { min, max }, encoding: 'DEFAULT',
    attributes: [
      { name: 'position', description: '', size: 12, numElements: 3, elementSize: 4, type: 'int32', min, max },
      { name: 'rgb', description: '', size: 6, numElements: 3, elementSize: 2, type: 'uint16', min: [0, 0, 0], max: [65535, 65535, 65535] },
      { name: 'classification', description: '', size: 1, numElements: 1, elementSize: 1, type: 'uint8', min: [0], max: [3] },
      { name: 'confidence', description: '', size: 4, numElements: 1, elementSize: 4, type: 'float', min: [0], max: [1] },
      { name: 'status', description: '', size: 1, numElements: 1, elementSize: 1, type: 'uint8', min: [0], max: [4] },
      { name: 'mv_votes', description: '', size: 1, numElements: 1, elementSize: 1, type: 'uint8', min: [0], max: [8] },
    ],
  }
  return { metadata, hier, buf, points: pts.length }
}
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296 } }
const OCTREE = buildOctree()

// ── HTTP ────────────────────────────────────────────────────────────────
const json = (res, body, status = 200) => { const s = JSON.stringify(body); res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(s), 'Access-Control-Allow-Origin': '*' }); res.end(s) }
const readBody = req => new Promise(resolve => { let d = ''; req.on('data', c => { d += c }); req.on('end', () => { try { resolve(d ? JSON.parse(d) : {}) } catch { resolve({}) } }) })

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost')
  const p = url.pathname
  const body = req.method === 'POST' || req.method === 'PATCH' || req.method === 'PUT' ? await readBody(req) : {}
  if (p === '/__mock/scenario') { scenario = body.name || 'sessions'; return json(res, { scenario }) }
  if (p === '/health') return json(res, { ok: true })
  if (p === '/api/auth/me') return json(res, { user: USER })
  if (p === '/api/auth/login') return json(res, { token: 'mock-token', user: USER })
  if (p === '/api/auth/users') return json(res, { users: USERS })
  if (p === '/sessions' && req.method === 'GET') return json(res, sessions())
  if (p === '/sessions' && req.method === 'POST') return json(res, { session_id: body.name })
  let m
  if ((m = p.match(/^\/sessions\/([^/]+)\/scans$/))) return json(res, { scans: scenario === 'empty' ? [] : (SCANS[m[1]] || []) })
  if ((m = p.match(/^\/sessions\/([^/]+)$/))) return json(res, { ok: true })
  if (p === '/api/pipelines/active') return json(res, { pipelines: scenario === 'pipeline' ? { 'tunel-norte': PIPELINE } : {} })
  if (p.startsWith('/api/certify/state/')) return json(res, scenario === 'certify' ? { acta: ACTA, pending_epochs: 2, epoch: 2, witness_fields: true, running_task: null } : { acta: null, pending_epochs: 0, epoch: 0, witness_fields: scenario === 'certify' })
  if (p.startsWith('/api/certify/epochs/')) return json(res, scenario === 'certify' ? { epoch: 2, previous: [{ epoch: 1, potree: true }, { epoch: 0, potree: false }] } : { epoch: 0, previous: [] })
  if (p.startsWith('/api/certify/edges/')) return json(res, scenario === 'certify' ? EDGES : { positions: [], odometry: [], loops: [], duplicates: [] })
  if (p.startsWith('/api/certify/attention/')) return json(res, scenario === 'certify' ? ATTENTION : { n: 0, items: [] })
  if (p.startsWith('/api/certify/report/')) return json(res, scenario === 'certify' ? REPORT : null, scenario === 'certify' ? 200 : 404)
  if (p.startsWith('/api/certify/')) return json(res, { ok: true, epoch: 2 })
  if (p.startsWith('/api/correction/state/')) return json(res, scenario === 'correction' ? {
    status: 'pending', epoch: 1, kind: 'objects', report: { epoch_to: 1, kind: 'objects', points_moved: 412_000, operator: 'hbarreto',
      solutions: [{ kf_span: [118, 131], rot_deg: 0.42, t_norm_m: 0.084, k: 1, residual_cm: { before: 8.4, after: 1.1 }, dof: ['yaw', 'tx', 'tz'] }],
      gates: [{ name: 'object_collapse', passed: true, detail: 'centroids 1.1 cm' }, { name: 'plausibility', passed: true, detail: 'rot 0.42° |t| 0.084 m' }, { name: 'scale_vs_da3', passed: false, advisory: true, detail: '1.6 % vs anchors' }],
      distribution: { keyframes_warped: 214, identity_until_kf: 118, max_step_between_keyframes_mm: 0.9, max_step_between_keyframes_deg: 0.01 } } } : { status: 'idle', epoch: scenario === 'certify' ? 2 : 0 })
  if (p.startsWith('/api/correction/ledger/')) return json(res, { entries: scenario === 'correction' ? [{ correction_id: 'c1', kind: 'objects', epoch_from: 0, epoch_to: 1, operator: 'hbarreto', created_at: '2026-09-14T09:00:00Z', verdict: 'pending', overrides: {} }] : [] })
  if (p.startsWith('/api/correction/artifacts/')) return json(res, { artifacts: scenario === 'correction' ? [{ path: 'output/tsdf/wall_1', kind: 'mesh', name: 'wall_1', geometry_epoch: 0, stale: true }] : [] })
  if (p.startsWith('/api/tasks/')) return json(res, { tasks: [] })
  if ((m = p.match(/^\/api\/sessions\/([^/]+)\/segmentation$/))) return json(res, { instances: hasCloud() && m[1] === 'tunel-norte' ? INSTANCES : [], unsegmented_points: 12_300_000 })
  if (p.startsWith('/api/sessions/') && p.endsWith('/prefs')) return json(res, {})
  if (p.startsWith('/api/sessions/') && p.endsWith('/sabana/exists')) return json(res, { exists: false })
  if (p.startsWith('/api/sessions/') && p.endsWith('/sabana/meta')) return json(res, SABANA_META)
  if (p.startsWith('/api/sessions/') && p.endsWith('/flythrough')) return json(res, { poses: [] })
  if (p.startsWith('/api/semantic/status')) return json(res, { status: 'up' })
  if (p === '/api/teams') return json(res, { teams: TEAMS })
  if (p === '/api/teams/available-users') return json(res, { users: USERS })
  if (p.endsWith('/messages')) return json(res, { messages: MESSAGES })
  if (p.endsWith('/activity')) return json(res, { activity: ACTIVITY })
  if (p === '/api/scene/objects') return json(res, { objects: [] })
  if (p === '/api/scene/volumes/list') return json(res, { volumes: [] })
  if (p.startsWith('/api/scene/')) return json(res, { ok: true, free_fraction: 1 })
  if (p.startsWith('/api/segmentation/shape/list') || p.startsWith('/api/segmentation/tsdf/list')) return json(res, { shapes: [] })
  if (p.startsWith('/api/segmentation/scene/')) return json(res, { elements: [] })
  if (p.startsWith('/api/segmentation/floor_level/')) return json(res, { candidates: hasCloud() ? [{ instance_id: 3, label: 'floor', height_m: 0.0 }] : [], selected: 3 })
  if (p === '/api/segmentation/level_floor') return json(res, { leveled: false, candidates: [] })
  if (p.startsWith('/api/segmentation/tsdf/status')) return json(res, { ok: true, instances: INSTANCES.map(i => ({ id: i.id, has_mesh: i.id === 1 })) })
  if (p.startsWith('/api/segmentation/')) return json(res, { ok: true })
  if (p.startsWith('/api/objects/scene/')) return json(res, { objects: [] })
  if (p === '/api/objects/library') return json(res, { items: [{ source: 'collection', name: 'durmiente_tipo.glb', url: '/static/durmiente.glb', size_mb: 1.2 }, { source: 'project', session: 'estacion-sur', name: 'banco_poisson.glb', url: '/static/banco.glb', size_mb: 4.8 }] })
  if (p.startsWith('/api/bim/auto_match/')) return json(res, { matches: COMPARE.results.map(r => ({ segment_label: r.label, element_key: r.element_key, ifc_type: r.ifc_type })) })
  if (p === '/api/bim/compare') return json(res, COMPARE)
  if (p.startsWith('/api/project/')) return json(res, { ok: true, scans: {} })
  if (p.startsWith('/api/spatial_qa')) return json(res, { answer: 'La holgura mínima entre la columna y el muro es de 1.72 m (medida con get_clearance).', tool_trace: [{ tool: 'get_clearance', arguments: {}, result: { clearance_m: 1.72 } }] })
  if (p === '/api/ar/sessions') return json(res, { sessions: [] })
  // Potree files
  if (p === '/potree/mock/metadata.json') return json(res, OCTREE.metadata)
  if (p === '/potree/mock/hierarchy.bin') { res.writeHead(200, { 'Content-Type': 'application/octet-stream', 'Content-Length': OCTREE.hier.length }); return res.end(OCTREE.hier) }
  if (p === '/potree/mock/octree.bin') {
    const range = req.headers.range
    if (range) {
      const [a, b] = range.replace('bytes=', '').split('-').map(Number)
      const end = Number.isFinite(b) ? b : OCTREE.buf.length - 1
      const slice = OCTREE.buf.subarray(a, end + 1)
      res.writeHead(206, { 'Content-Type': 'application/octet-stream', 'Content-Length': slice.length, 'Content-Range': `bytes ${a}-${end}/${OCTREE.buf.length}` })
      return res.end(slice)
    }
    res.writeHead(200, { 'Content-Type': 'application/octet-stream', 'Content-Length': OCTREE.buf.length }); return res.end(OCTREE.buf)
  }
  if (p.startsWith('/potree_epoch/')) { // the "before" layer reuses the same octree
    const sub = p.split('/').slice(4).join('/')
    if (sub === 'metadata.json') return json(res, OCTREE.metadata)
    if (sub === 'hierarchy.bin') { res.writeHead(200, { 'Content-Type': 'application/octet-stream' }); return res.end(OCTREE.hier) }
    if (sub === 'octree.bin') { res.writeHead(200, { 'Content-Type': 'application/octet-stream' }); return res.end(OCTREE.buf) }
  }
  json(res, { detail: `mock: no fixture for ${req.method} ${p}` }, 404)
})

// ── WebSocket (RFC 6455, text frames only) ──────────────────────────────
function wsSend(sock, obj) {
  const payload = Buffer.from(JSON.stringify(obj))
  let header
  if (payload.length < 126) header = Buffer.from([0x81, payload.length])
  else if (payload.length < 65536) { header = Buffer.alloc(4); header[0] = 0x81; header[1] = 126; header.writeUInt16BE(payload.length, 2) }
  else { header = Buffer.alloc(10); header[0] = 0x81; header[1] = 127; header.writeBigUInt64BE(BigInt(payload.length), 2) }
  if (!sock.destroyed) sock.write(Buffer.concat([header, payload]))
}
function wsParse(buf, onText) {
  let off = 0
  while (off + 2 <= buf.length) {
    const fin = buf[off] & 0x80, opcode = buf[off] & 0x0f, masked = buf[off + 1] & 0x80
    let len = buf[off + 1] & 0x7f, pos = off + 2
    if (len === 126) { len = buf.readUInt16BE(pos); pos += 2 } else if (len === 127) { len = Number(buf.readBigUInt64BE(pos)); pos += 8 }
    const mask = masked ? buf.subarray(pos, pos + 4) : null
    if (masked) pos += 4
    if (pos + len > buf.length) return buf.subarray(off)
    const data = Buffer.from(buf.subarray(pos, pos + len))
    if (mask) for (let i = 0; i < data.length; i++) data[i] ^= mask[i % 4]
    if (opcode === 1 && fin) onText(data.toString('utf8'))
    if (opcode === 8) return null
    off = pos + len
  }
  return Buffer.alloc(0)
}
server.on('upgrade', (req, sock) => {
  const key = req.headers['sec-websocket-key']
  const accept = crypto.createHash('sha1').update(key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64')
  sock.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`)
  const path = new URL(req.url, 'http://localhost').pathname
  let pending = Buffer.alloc(0)
  sock.on('data', chunk => {
    pending = Buffer.concat([pending, chunk])
    const rest = wsParse(pending, text => {
      let msg = {}
      try { msg = JSON.parse(text) } catch { return }
      if (path === '/ws/viewer' && msg.type === 'load_session' && hasCloud()) {
        setTimeout(() => {
          wsSend(sock, { type: 'potree_ready', url: '/potree/mock/', points: OCTREE.points, session_id: msg.session_id, hasConfidence: true, floorTransform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] })
          if (scenario === 'pipeline') wsSend(sock, { type: 'pipeline_progress', ...PIPELINE })
        }, 200)
      }
      if (path === '/ws/viewer' && msg.type === 'run_pipeline') wsSend(sock, { type: 'pipeline_progress', ...PIPELINE, session_id: msg.session_id })
    })
    if (rest === null) { sock.end(); return }
    pending = rest
  })
  sock.on('error', () => {})
  if (path === '/ws/logs') {
    const lines = [['info', 'pipeline[tunel-norte] stage vlm: 214/512 keyframes described'], ['warn', 'salad: 3 candidates below threshold, kept as advisory'], ['info', 'certify: iteration 1 objective 0.318 -> 0.142'], ['error', 'fuse: pair "banco" dropped (size ratio 1.21)'], ['debug', 'ws viewer: potree_ready sent (43,800 pts)']]
    lines.forEach((l, i) => setTimeout(() => wsSend(sock, { ts: `09:1${i}:0${i}`, level: l[0], msg: l[1] }), 100 * i))
  }
  if (path === '/ws/team') setTimeout(() => wsSend(sock, { type: 'presence_update', online: [{ user_id: 2, username: 'topografia', task: 'segmentando tunel-norte' }] }), 100)
})

server.listen(PORT, () => console.log(`mock backend on http://localhost:${PORT} (scenario: ${scenario})`))
