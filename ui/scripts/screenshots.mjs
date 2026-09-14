#!/usr/bin/env node
/**
 * screenshots.mjs — before/after gallery (prompt_ui.txt §11, §12).
 *
 * Starts the mock backend (scripts/mock-server.mjs), a Vite dev server for
 * the working tree ("after") and, unless --after-only, a second Vite dev
 * server from a git worktree of the base commit ("before", proxy patched to
 * the mock). Captures every view/state of §12 at 1440×900 and 1024×768, in
 * both densities (before has no density switch: captured once per size),
 * into ui/design-review/<view>__<state>__<density>__<before|after>.png plus
 * contact-sheet.png with the whole set in a grid.
 *
 * Usage: node scripts/screenshots.mjs [--after-only] [--base <commit>]
 */
import { chromium } from 'playwright'
import { spawn, execSync } from 'node:child_process'
import { mkdirSync, existsSync, readdirSync, rmSync, writeFileSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createServer } from 'node:http'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const UI = path.resolve(__dirname, '..')
const OUT = path.join(UI, 'design-review')
const args = process.argv.slice(2)
const AFTER_ONLY = args.includes('--after-only')
const BASE = args.includes('--base') ? args[args.indexOf('--base') + 1] : '506acd0'
const MOCK_PORT = 8790, AFTER_PORT = 5199, BEFORE_PORT = 5198
const SIZES = [{ w: 1440, h: 900 }, { w: 1024, h: 768 }]
const DENSITIES = ['compact', 'comfortable']
const es = JSON.parse(readFileSync(path.join(UI, 'src/i18n/es.json'), 'utf8'))
const t = key => key.split('.').reduce((o, k) => (o ? o[k] : undefined), es) ?? key
const glossary = { session: 'sesión', sessions: 'sesiones', Session: 'Sesión', Sessions: 'Sesiones', scan: 'escaneo', scans: 'escaneos', Scan: 'Escaneo', Scans: 'Escaneos', instance: 'instancia', instances: 'instancias', Instance: 'Instancia', Instances: 'Instancias', correction: 'corrección', Correction: 'Corrección', acta: 'acta', Acta: 'Acta', fusion: 'fusión', Fusion: 'Fusión', witness: 'testigo', deviation: 'desviación', Deviation: 'Desviación', tolerance: 'tolerancia', Tolerance: 'Tolerancia', epoch: 'época geométrica', Epoch: 'Época geométrica' }
const T = (key, vars = {}) => t(key).replace(/\{(@?\w+)\}/g, (m, k) => (k.startsWith('@') ? glossary[k.slice(1)] ?? m : vars[k] ?? m))

mkdirSync(OUT, { recursive: true })
for (const f of readdirSync(OUT)) rmSync(path.join(OUT, f))

const procs = []
const spawnLogged = (cmd, cmdArgs, opts) => { const p = spawn(cmd, cmdArgs, { ...opts, stdio: ['ignore', 'pipe', 'pipe'] }); p.stdout.on('data', () => {}); p.stderr.on('data', () => {}); procs.push(p); return p }
const waitHttp = async (url, ms = 60000) => { const t0 = Date.now(); for (;;) { try { const r = await fetch(url); if (r.ok || r.status === 404) return } catch { /* not yet */ } if (Date.now() - t0 > ms) throw new Error(`timeout waiting ${url}`); await new Promise(r => setTimeout(r, 300)) } }
const portFree = port => new Promise(res => { const s = createServer(); s.once('error', () => res(false)); s.listen(port, () => s.close(() => res(true))) })

// ── servers ─────────────────────────────────────────────────────────────
if (await portFree(MOCK_PORT)) spawnLogged('node', ['scripts/mock-server.mjs', String(MOCK_PORT)], { cwd: UI })
await waitHttp(`http://localhost:${MOCK_PORT}/health`)
const viteEnv = { ...process.env, WEB_ONLY: 'true', STAC_API_TARGET: `http://localhost:${MOCK_PORT}` }
spawnLogged('npx', ['vite', '--port', String(AFTER_PORT), '--strictPort'], { cwd: UI, env: viteEnv })
await waitHttp(`http://localhost:${AFTER_PORT}/`)

let beforeDir = null
if (!AFTER_ONLY) {
  beforeDir = path.join('/tmp', 'stac-ui-before')
  if (existsSync(beforeDir)) { try { execSync(`git -C ${UI} worktree remove --force ${beforeDir}`, { stdio: 'ignore' }) } catch { rmSync(beforeDir, { recursive: true, force: true }) } }
  execSync(`git -C ${UI} worktree add --detach ${beforeDir} ${BASE}`, { stdio: 'ignore' })
  const beforeUi = path.join(beforeDir, 'ui')
  const cfg = path.join(beforeUi, 'vite.config.ts')
  writeFileSync(cfg, readFileSync(cfg, 'utf8').replace(/https:\/\/localhost:8765/g, `http://localhost:${MOCK_PORT}`))
  execSync(`ln -sfn ${path.join(UI, 'node_modules')} ${path.join(beforeUi, 'node_modules')}`)
  spawnLogged('npx', ['vite', '--port', String(BEFORE_PORT), '--strictPort'], { cwd: beforeUi, env: { ...process.env, WEB_ONLY: 'true' } })
  await waitHttp(`http://localhost:${BEFORE_PORT}/`)
}

const setScenario = name => fetch(`http://localhost:${MOCK_PORT}/__mock/scenario`, { method: 'POST', body: JSON.stringify({ name }) })
const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] })
const shots = []

async function openApp(page, port, { density, loggedIn = true, layout = {} } = {}) {
  await page.addInitScript(({ density, loggedIn, layout }) => {
    localStorage.clear()
    if (loggedIn) localStorage.setItem('stac_token', 'mock-token')
    localStorage.setItem('stac.lang', 'es')
    localStorage.setItem('stac.layout.v1', JSON.stringify({ leftTab: 'sessions', leftWidth: 280, inspectorOpen: true, inspectorTab: 'assistant', inspectorWidth: 320, dockOpen: false, dockTab: 'console', dockHeight: 200, density, ...layout }))
    localStorage.setItem('stac.assistantOpen', '1')
  }, { density, loggedIn, layout })
  await page.goto(`http://localhost:${port}/`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(600)
  if (loggedIn) { await clickIf(page.getByRole('button', { name: T('welcome.connect') }).first()); await page.waitForTimeout(700) }
}

async function shot(page, name, state, density, phase) {
  const file = `${name}__${state}__${density}__${phase}.png`
  // Park the pointer on the brand block (no hover affordance there) and wait
  // out the tooltip delay: otherwise the tooltip of whatever control the
  // capture last clicked stays open and covers the panel underneath.
  await page.mouse.move(8, 8)
  await page.waitForTimeout(250)
  await page.screenshot({ path: path.join(OUT, file) })
  shots.push(file)
}

const byLabel = (page, key) => page.getByLabel(T(key), { exact: true }).first()
const byText = (page, key, vars) => page.getByText(T(key, vars), { exact: true }).first()
const clickIf = async loc => { try { await loc.click({ timeout: 2500 }); return true } catch { return false } }
const loadSession = async page => { await clickIf(byLabel(page, 'sessions.load')); await page.waitForTimeout(3500) }

// ── AFTER captures ──────────────────────────────────────────────────────
for (const size of SIZES) {
  for (const density of DENSITIES) {
    const tag = `${density}-${size.w}`
    const ctx = await browser.newContext({ viewport: { width: size.w, height: size.h }, locale: 'es-MX' })
    const page = await ctx.newPage()
    page.on('pageerror', e => console.error(`[after ${tag}] page error:`, e.message))

    await setScenario('sessions')
    await openApp(page, AFTER_PORT, { density, loggedIn: false })
    await shot(page, 'login', 'default', tag, 'after')

    await setScenario('empty')
    await openApp(page, AFTER_PORT, { density })
    await shot(page, 'sessions', 'empty', tag, 'after')

    await setScenario('sessions')
    await openApp(page, AFTER_PORT, { density })
    await shot(page, 'sessions', 'list', tag, 'after')
    if (await clickIf(byText(page, 'sessions.new'))) {
      await page.keyboard.type('galeria-demo')
      await page.keyboard.press('Enter')
      await page.waitForTimeout(700)
      await shot(page, 'toast', 'session-created', tag, 'after')
      await page.waitForTimeout(200)
    }
    await clickIf(byText(page, 'menu.settings'))
    await page.keyboard.press('Escape')
    await clickIf(page.getByLabel(T('menu.settings'), { exact: true }).last())
    await page.waitForTimeout(300)
    await shot(page, 'settings', 'density-language', tag, 'after')
    await page.keyboard.press('Escape')
    await page.keyboard.press('Control+K')
    await page.waitForTimeout(300)
    await page.keyboard.type('med')
    await page.waitForTimeout(200)
    await shot(page, 'palette', 'search', tag, 'after')
    await page.keyboard.press('Escape')
    await clickIf(byLabel(page, 'sessions.delete'))
    await page.waitForTimeout(300)
    await shot(page, 'dialog', 'destructive', tag, 'after')
    await page.keyboard.press('Escape')
    await clickIf(page.getByRole('button', { name: T('menu.userManagement') }).first())
    if (!(await page.getByText(T('admin.title'), { exact: true }).first().isVisible().catch(() => false))) {
      await clickIf(page.getByRole('button', { name: /hbarreto/ }).first())
      await clickIf(byText(page, 'menu.userManagement'))
    }
    await page.waitForTimeout(400)
    await shot(page, 'admin', 'users', tag, 'after')
    await clickIf(page.getByRole('tab', { name: new RegExp(T('admin.teams')) }))
    await page.waitForTimeout(300)
    await shot(page, 'admin', 'teams', tag, 'after')
    await page.keyboard.press('Escape')

    await setScenario('pipeline')
    await openApp(page, AFTER_PORT, { density })
    await loadSession(page)
    await shot(page, 'sessions', 'pipeline-running', tag, 'after')
    await clickIf(page.getByRole('tab', { name: T('dock.jobs') }))
    await clickIf(byLabel(page, 'dock.jobs'))
    await page.keyboard.press('Control+`')
    await page.waitForTimeout(300)
    await clickIf(page.getByRole('tab', { name: new RegExp(T('dock.jobs')) }))
    await page.waitForTimeout(300)
    await shot(page, 'dock', 'jobs', tag, 'after')

    await setScenario('cloud')
    await openApp(page, AFTER_PORT, { density })
    await shot(page, 'sessions', 'loading-splash', tag, 'after')
    await loadSession(page)
    await shot(page, 'viewport', 'cloud', tag, 'after')
    await clickIf(byLabel(page, 'instances.title'))
    await page.waitForTimeout(400)
    await shot(page, 'instances', 'populated', tag, 'after')
    await clickIf(byLabel(page, 'sessions.title'))
    await clickIf(page.getByRole('treeitem').filter({ hasText: 'estacion-sur' }).first())
    await clickIf(byLabel(page, 'instances.title'))
    await page.waitForTimeout(400)
    await shot(page, 'instances', 'no-segmentation', tag, 'after')
    await clickIf(byLabel(page, 'sessions.title'))
    await clickIf(page.getByRole('treeitem').filter({ hasText: 'tunel-norte' }).first())
    await clickIf(byLabel(page, 'instances.title'))
    await page.waitForTimeout(400)
    await clickIf(page.getByRole('treeitem').filter({ hasText: 'column' }).first())
    await page.waitForTimeout(300)
    await shot(page, 'instances', 'selected-properties', tag, 'after')
    await clickIf(byLabel(page, 'toolbar.measureDistance'))
    const canvas = page.locator('.viewport-canvas')
    const box = await canvas.boundingBox()
    if (box) {
      await page.mouse.click(box.x + box.width * 0.45, box.y + box.height * 0.55)
      await page.waitForTimeout(200)
      await page.mouse.click(box.x + box.width * 0.62, box.y + box.height * 0.58)
      await page.waitForTimeout(400)
    }
    await shot(page, 'viewport', 'measure-reading', tag, 'after')
    await clickIf(byLabel(page, 'toolbar.navigate'))
    await clickIf(byLabel(page, 'toolbar.colorBy'))
    await clickIf(page.getByRole('radio', { name: T('colorMode.status') }))
    await page.waitForTimeout(400)
    await shot(page, 'viewport', 'color-status', tag, 'after')
    await page.keyboard.press('Escape')
    await clickIf(byLabel(page, 'toolbar.brush'))
    await page.waitForTimeout(300)
    await shot(page, 'viewport', 'brush-panel', tag, 'after')
    await clickIf(byLabel(page, 'toolbar.navigate'))
    await clickIf(byLabel(page, 'scans.title'))
    await page.waitForTimeout(300)
    await shot(page, 'scans', 'list', tag, 'after')
    await clickIf(page.getByRole('button', { name: T('instances.fuse'), exact: true }).first())
    await page.waitForTimeout(800)
    await shot(page, 'scans', 'fuse-modal', tag, 'after')
    await page.keyboard.press('Escape')
    await clickIf(byLabel(page, 'instances.title'))
    await clickIf(page.getByRole('button', { name: T('instances.meshing'), exact: true }).first())
    await page.waitForTimeout(500)
    await shot(page, 'meshing', 'dialog', tag, 'after')
    await page.keyboard.press('Escape')
    await clickIf(page.getByRole('button', { name: T('instances.correction'), exact: true }).first())
    await page.waitForTimeout(500)
    await shot(page, 'correction', 'dialog', tag, 'after')
    await page.keyboard.press('Escape')
    await clickIf(byLabel(page, 'bim.title'))
    await page.waitForTimeout(300)
    await shot(page, 'bim', 'empty', tag, 'after')
    await clickIf(byLabel(page, 'team.title'))
    await page.waitForTimeout(500)
    await shot(page, 'team', 'members', tag, 'after')
    await clickIf(page.getByRole('tab', { name: new RegExp(T('team.chat')) }))
    await page.waitForTimeout(300)
    await shot(page, 'team', 'chat', tag, 'after')
    await page.keyboard.press('Control+`')
    await page.waitForTimeout(700)
    await shot(page, 'dock', 'console', tag, 'after')
    await clickIf(page.getByRole('tab', { name: new RegExp(T('dock.timeline')) }))
    await page.waitForTimeout(300)
    await shot(page, 'dock', 'timeline', tag, 'after')
    await page.keyboard.press('Control+`')
    await clickIf(page.getByRole('tab', { name: new RegExp(T('assistant.title')) }))
    await page.fill('input[placeholder="' + T('assistant.placeholder') + '"]', T('assistant.suggestions.clearance')).catch(() => {})
    await page.keyboard.press('Enter')
    await page.waitForTimeout(1200)
    await shot(page, 'assistant', 'answer', tag, 'after')

    await setScenario('certify')
    await openApp(page, AFTER_PORT, { density, layout: { inspectorTab: 'acta' } })
    await loadSession(page)
    await page.waitForTimeout(800)
    await shot(page, 'certify', 'kit-attention', tag, 'after')
    await clickIf(page.getByRole('tab', { name: new RegExp(T('edges.title')) }))
    await page.waitForTimeout(300)
    await shot(page, 'certify', 'edges', tag, 'after')
    await clickIf(page.getByRole('tab', { name: new RegExp(T('inspector.acta')) }).last())
    await page.waitForTimeout(300)
    await shot(page, 'certify', 'acta', tag, 'after')
    await shot(page, 'viewport', 'trajectory-edges', tag, 'after')
    await clickIf(byLabel(page, 'toolbar.colorBy'))
    await clickIf(page.getByRole('radio', { name: T('colorMode.votes') }))
    await page.waitForTimeout(400)
    await shot(page, 'viewport', 'color-votes-legend', tag, 'after')
    await page.keyboard.press('Escape')

    await setScenario('correction')
    await openApp(page, AFTER_PORT, { density })
    await loadSession(page)
    await page.waitForTimeout(600)
    await shot(page, 'correction', 'verdict-dialog', tag, 'after')

    await setScenario('bim')
    await openApp(page, AFTER_PORT, { density, layout: { dockOpen: true, dockTab: 'report' } })
    await loadSession(page)
    await clickIf(page.getByRole('button', { name: T('deviation.runComparison'), exact: true }).first())
    await page.waitForTimeout(1200)
    await shot(page, 'report', 'deviation', tag, 'after')

    await ctx.close()
  }
}

// ── BEFORE captures (base commit, one density) ──────────────────────────
if (beforeDir) {
  for (const size of SIZES) {
    const tag = `compact-${size.w}`
    const ctx = await browser.newContext({ viewport: { width: size.w, height: size.h }, locale: 'es-MX' })
    const page = await ctx.newPage()
    const openBefore = async (loggedIn = true) => {
      await page.addInitScript(({ loggedIn }) => { localStorage.clear(); if (loggedIn) localStorage.setItem('stac_token', 'mock-token'); localStorage.setItem('stac.assistantOpen', '1') }, { loggedIn })
      await page.goto(`http://localhost:${BEFORE_PORT}/`, { waitUntil: 'networkidle' })
      await page.waitForTimeout(600)
    }
    await setScenario('sessions'); await openBefore(false); await shot(page, 'login', 'default', tag, 'before')
    await setScenario('empty'); await openBefore(); await clickIf(page.getByText('Connect to Server').first()); await page.waitForTimeout(500); await shot(page, 'sessions', 'empty', tag, 'before')
    await setScenario('sessions'); await openBefore(); await clickIf(page.getByText('Connect to Server').first()); await page.waitForTimeout(500); await shot(page, 'sessions', 'list', tag, 'before')
    await clickIf(page.getByText('hbarreto').first()); await clickIf(page.getByText('User Management').first()); await page.waitForTimeout(400); await shot(page, 'admin', 'users', tag, 'before'); await page.keyboard.press('Escape')
    await clickIf(page.getByTitle('Delete Project').first()); await page.waitForTimeout(300); await shot(page, 'dialog', 'destructive', tag, 'before'); await page.keyboard.press('Escape')
    await setScenario('pipeline'); await openBefore(); await clickIf(page.getByText('Connect to Server').first()); await page.waitForTimeout(300); await clickIf(page.getByTitle('Load Session').first()); await page.waitForTimeout(3500); await shot(page, 'sessions', 'pipeline-running', tag, 'before')
    await setScenario('cloud'); await openBefore(); await clickIf(page.getByText('Connect to Server').first()); await page.waitForTimeout(300); await clickIf(page.getByTitle('Load Session').first()); await page.waitForTimeout(3500); await shot(page, 'viewport', 'cloud', tag, 'before')
    await clickIf(page.getByTitle('Segments').first()); await page.waitForTimeout(300); await shot(page, 'instances', 'populated', tag, 'before')
    await clickIf(page.getByTitle('Measure Distance (M)').first())
    const c = await page.locator('.viewport-canvas').boundingBox()
    if (c) { await page.mouse.click(c.x + c.width * 0.45, c.y + c.height * 0.55); await page.waitForTimeout(200); await page.mouse.click(c.x + c.width * 0.62, c.y + c.height * 0.58); await page.waitForTimeout(400) }
    await shot(page, 'viewport', 'measure-reading', tag, 'before')
    await clickIf(page.getByTitle('Mark zones (right-click) to erase or reassign').first()); await page.waitForTimeout(300); await shot(page, 'viewport', 'brush-panel', tag, 'before')
    await clickIf(page.getByTitle('BIM Navigator').first()); await page.waitForTimeout(300); await shot(page, 'bim', 'empty', tag, 'before')
    await clickIf(page.getByTitle('Team').first()); await page.waitForTimeout(500); await shot(page, 'team', 'members', tag, 'before')
    await clickIf(page.getByTitle('Console').first()); await page.waitForTimeout(700); await shot(page, 'dock', 'console', tag, 'before')
    await clickIf(page.getByTitle('Segments').first()); await clickIf(page.getByText('Meshing').first()); await page.waitForTimeout(500); await shot(page, 'meshing', 'dialog', tag, 'before'); await page.keyboard.press('Escape')
    await clickIf(page.getByText('Correction').first()); await page.waitForTimeout(500); await shot(page, 'correction', 'dialog', tag, 'before'); await page.keyboard.press('Escape')
    await clickIf(page.getByText('Fuse').first()); await page.waitForTimeout(800); await shot(page, 'scans', 'fuse-modal', tag, 'before'); await page.keyboard.press('Escape')
    await page.fill('input[placeholder="Ask about the scene…"]', 'What is the clearance between the two nearest walls?').catch(() => {}); await page.keyboard.press('Enter'); await page.waitForTimeout(1200); await shot(page, 'assistant', 'answer', tag, 'before')
    await setScenario('certify'); await openBefore(); await clickIf(page.getByText('Connect to Server').first()); await page.waitForTimeout(300); await clickIf(page.getByTitle('Load Session').first()); await page.waitForTimeout(4000); await shot(page, 'certify', 'kit-attention', tag, 'before')
    await setScenario('correction'); await openBefore(); await clickIf(page.getByText('Connect to Server').first()); await page.waitForTimeout(300); await clickIf(page.getByTitle('Load Session').first()); await page.waitForTimeout(3500); await shot(page, 'correction', 'verdict-dialog', tag, 'before')
    await ctx.close()
  }
}

// ── contact sheet ───────────────────────────────────────────────────────
{
  const files = readdirSync(OUT).filter(f => f.endsWith('.png') && f !== 'contact-sheet.png').sort()
  const cols = 4, thumbW = 360, thumbH = 225, label = 22
  const rows = Math.ceil(files.length / cols)
  const html = `<!doctype html><html><body style="margin:0;background:#0a0d14;font-family:Inter,sans-serif;">
  <div style="display:grid;grid-template-columns:repeat(${cols},${thumbW}px);gap:12px;padding:12px;width:${cols * (thumbW + 12) + 12}px">
  ${files.map(f => `<figure style="margin:0"><img src="file://${path.join(OUT, f)}" style="width:${thumbW}px;height:${thumbH}px;object-fit:cover;object-position:top left;border:1px solid #262d3a;display:block"><figcaption style="color:#a4adba;font:11px monospace;height:${label}px;overflow:hidden">${f}</figcaption></figure>`).join('')}
  </div></body></html>`
  const sheetPath = path.join(OUT, 'contact-sheet.html')
  writeFileSync(sheetPath, html)
  const ctx = await browser.newContext({ viewport: { width: cols * (thumbW + 12) + 24, height: Math.min(16000, rows * (thumbH + label + 12) + 24) } })
  const page = await ctx.newPage()
  await page.goto(`file://${sheetPath}`)
  await page.waitForTimeout(500)
  await page.screenshot({ path: path.join(OUT, 'contact-sheet.png'), fullPage: true })
  await ctx.close()
  rmSync(sheetPath)
}

await browser.close()
for (const p of procs) p.kill()
if (beforeDir) { try { execSync(`git -C ${UI} worktree remove --force ${beforeDir}`, { stdio: 'ignore' }) } catch { /* leave it */ } }
console.log(`screenshots: ${shots.length} captures in ${path.relative(process.cwd(), OUT)} + contact-sheet.png`)
process.exit(0)
